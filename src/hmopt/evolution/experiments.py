"""Operator-configured business build/test adapter with durable exclusive resource claims.

The adapter receives exact candidate/manifest/policy JSON and writes report JSON.
The platform runs argv without a shell; adapter success does not bypass validation.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .correctness import CorrectnessReport
from .store import ConflictError, canonical_json, digest
from .validation import ABReport


def prepare_experiment(service, candidate_id, config, *, actor, request_id):
    actor = service._actor(actor)
    request = {
        "candidate_id": candidate_id,
        "config": config.model_dump(mode="json"),
        "actor": actor,
    }
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        row = service.store.get(db, "candidate", candidate_id)
        data = row["data"]
        if (
            data["stage"] != "code_approved"
            or data.get("validation")
            or actor == data["implementation"]["actor"]
        ):
            raise ValueError(
                "Experiment requires an independent validator and unsealed code-approved candidate"
            )
        service._assert_batch_scope(db, candidate_id, "validated")
        manifest_sha = data["candidate"].get("workspace_manifest_sha256")
        manifest = service.store.read_evidence(manifest_sha, _db=db) if manifest_sha else None
        frozen = {
            "candidate": row,
            "workspace_manifest": manifest,
            "execution_repo": service._execution_repo(
                data["candidate"], data.get("execution_context")
            ),
            "baseline_revision": data["candidate"]["repo_revision"],
            "implementation_revision": data["implementation"]["revision"],
            "policy": data["plan"]["validation"],
            "adapter": request["config"],
            "contract": "Build and execute both exact revisions. Keep all other manifest projects fixed. Produce report JSON; do not infer success from exit status or alter the approved policy.",
        }
        sha = service.store.evidence(db, frozen)
        key = "experiment_" + digest([request_id, request, sha])
        result = service.store.put(
            db,
            "experiment",
            key,
            {
                "candidate_id": candidate_id,
                "candidate_version": row["version"],
                "request_sha256": sha,
                "status": "prepared",
                "actor": actor,
                "resource_id": config.resource_id,
                "config_sha256": digest(request["config"]),
                "created_at": time.time(),
            },
        )
        service.store.remember(db, request_id, request, result)
        service.store.event(db, key, "experiment_prepared", actor, {"request_sha256": sha})
        return result


def run_experiment(service, experiment_id, config):
    """Explicit operational action; never called by discovery or the default supervisor."""
    with service.store.transaction() as db:
        job = service.store.get(db, "experiment", experiment_id)
        data = job["data"]
        if data["status"] == "complete":
            return job
        if data["status"] != "prepared" or data["config_sha256"] != digest(
            config.model_dump(mode="json")
        ):
            raise ConflictError(
                "Experiment is already running/uncertain or adapter configuration changed"
            )
        candidate = service.store.get(db, "candidate", data["candidate_id"])
        if (
            candidate["version"] != data["candidate_version"]
            or candidate["data"]["stage"] != "code_approved"
            or candidate["data"].get("validation")
        ):
            raise ConflictError("Experiment candidate changed since preparation")
        service._assert_batch_scope(db, data["candidate_id"], "validated")
        if db.execute(
            "SELECT 1 FROM records WHERE kind='experiment_claim' AND id=?", (config.resource_id,)
        ).fetchone():
            raise ConflictError(
                "Build/device resource is already reserved; inspect its current experiment"
            )
        if db.execute(
            "SELECT 1 FROM records WHERE kind='experiment_candidate_claim' AND id=?",
            (data["candidate_id"],),
        ).fetchone():
            raise ConflictError("Candidate already has an active experiment")
        service.store.put(
            db, "experiment_candidate_claim", data["candidate_id"], {"experiment_id": experiment_id}
        )
        service.store.put(
            db, "experiment_claim", config.resource_id, {"experiment_id": experiment_id}
        )
        data.update(status="running", started_at=time.time())
        job = service.store.put(db, "experiment", experiment_id, data, job["version"])
    directory = service.store.root / "experiments" / experiment_id
    try:
        if directory.resolve() != directory:
            raise ValueError("Experiment artifact directory is redirected")
        directory.mkdir(parents=True, exist_ok=False)
        request_path, output_path = directory / "request.json", directory / "result.json"
        frozen = service.store.read_evidence(data["request_sha256"])
        request_path.write_text(canonical_json(frozen) + "\n", encoding="utf-8", newline="\n")
        cwd = Path(config.cwd)
        if cwd.resolve() != cwd or not cwd.is_dir():
            raise ValueError("Validation adapter cwd is missing or redirected")
        environment = dict(os.environ)
        environment.update(
            HMOPT_EXPERIMENT_REQUEST=str(request_path),
            HMOPT_EXPERIMENT_RESULT=str(output_path),
            HMOPT_EXPERIMENT_ID=experiment_id,
            HMOPT_TARGET_REPO=frozen["execution_repo"],
        )
        # stdout is not a result channel, and may contain credentials/device logs. The
        # adapter writes shareable evidence beneath its exact request directory instead.
        completed = subprocess.run(
            config.command,
            cwd=cwd,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=config.timeout_seconds,
            check=False,
        )
        if completed.returncode:
            raise ValueError(f"Validation adapter exited with code {completed.returncode}")
        if (
            not output_path.is_file()
            or output_path.resolve() != output_path
            or output_path.stat().st_size > 16 * 1024 * 1024
        ):
            raise ValueError("Adapter must write bounded regular result.json")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(result, dict) and set(result) == {"lmbench_manifest"}:
            from .lmbench import LmbenchManifest, convert_lmbench_report

            converted = convert_lmbench_report(
                service,
                data["candidate_id"],
                LmbenchManifest.model_validate(result["lmbench_manifest"]),
                artifacts_root=directory,
            )
            result = {"report": converted["report"], "raw_result": result}
        elif isinstance(result, dict) and set(result) == {"ic_compare", "ic_manifest"}:
            from .reports import ICManifest, convert_ic_report

            converted = convert_ic_report(
                service,
                data["candidate_id"],
                result["ic_compare"],
                ICManifest.model_validate(result["ic_manifest"]),
            )
            result = {"report": converted["report"], "raw_result": result}
        elif not isinstance(result, dict) or set(result) != {"report"}:
            raise ValueError(
                "Adapter result requires report, lmbench_manifest, or ic_compare + ic_manifest"
            )
        if not isinstance(result["report"], dict):
            raise ValueError("Adapter report must be a JSON object")  # noqa: TRY004
        report = (
            CorrectnessReport.model_validate(result["report"])
            if result["report"].get("kind") == "correctness"
            else ABReport.model_validate(result["report"])
        )
        with service.store.transaction() as db:
            current = service.store.get(db, "experiment", experiment_id)
            if current["version"] != job["version"] or current["data"]["status"] != "running":
                raise ConflictError("Experiment was retired or superseded; discard completion")
            sha = service.store.evidence(
                db,
                {
                    "request_sha256": data["request_sha256"],
                    "result": result,
                    "exit_code": completed.returncode,
                    "collected_at": time.time(),
                },
            )
            data["output_sha256"] = sha
            job = service.store.put(db, "experiment", experiment_id, data, job["version"])
        with service.store.transaction() as db:
            current = service.store.get(db, "experiment", experiment_id)
            if current["version"] != job["version"] or current["data"]["status"] != "running":
                raise ConflictError("Experiment was retired or superseded; discard completion")
            verdict = service.validate(
                data["candidate_id"],
                report,
                actor=data["actor"],
                expected_version=data["candidate_version"],
                request_id="experiment-result:" + experiment_id,
                _db=db,
            )
            data.update(status="complete", validation=verdict, completed_at=time.time())
            row = service.store.put(db, "experiment", experiment_id, data, job["version"])
            db.execute(
                "DELETE FROM records WHERE kind='experiment_claim' AND id=? AND json_extract(payload,'$.experiment_id')=?",
                (config.resource_id, experiment_id),
            )
            db.execute(
                "DELETE FROM records WHERE kind='experiment_candidate_claim' AND json_extract(payload,'$.experiment_id')=?",
                (experiment_id,),
            )
            service.store.event(
                db, experiment_id, "experiment_completed", data["actor"], {"output_sha256": sha}
            )
            return row
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        with service.store.transaction() as db:
            current = service.store.get(db, "experiment", experiment_id)
            if current["version"] == job["version"]:
                data.update(status="attention", error=str(error)[:1000])
                service.store.put(db, "experiment", experiment_id, data, job["version"])
        raise


def retire_experiment(service, experiment_id, *, actor, expected_version, execution_stopped):
    """Explicit acknowledgement covers adapter children/remote device tasks, not only a PID."""
    if execution_stopped is not True:
        raise ValueError(
            "Confirm adapter and remote/device work have stopped before releasing resource"
        )
    actor = service._actor(actor)
    with service.store.transaction() as db:
        row = service.store.get(db, "experiment", experiment_id)
        if row["version"] != expected_version or row["data"]["status"] not in {
            "running",
            "attention",
        }:
            raise ConflictError("Retirement requires current running/attention experiment")
        row["data"].update(status="retired", retired_by=actor)
        result = service.store.put(db, "experiment", experiment_id, row["data"], row["version"])
        db.execute(
            "DELETE FROM records WHERE kind='experiment_claim' AND json_extract(payload,'$.experiment_id')=?",
            (experiment_id,),
        )
        db.execute(
            "DELETE FROM records WHERE kind='experiment_candidate_claim' AND json_extract(payload,'$.experiment_id')=?",
            (experiment_id,),
        )
        service.store.event(
            db, experiment_id, "experiment_retired", actor, {"execution_stopped": True}
        )
        return result
