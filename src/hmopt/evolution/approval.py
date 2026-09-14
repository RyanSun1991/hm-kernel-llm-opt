"""Frozen approval requests, durable notifications, and gateway-authenticated decisions."""

from __future__ import annotations

import hashlib
import hmac
import smtplib
import ssl
import time
from email.message import EmailMessage
from typing import Literal
from urllib.parse import quote

from pydantic import Field

from .production import ApprovalConfig, Strict, secret
from .research import pattern_digest
from .store import ConflictError, digest
from .transport import request_json


class Decision(Strict):
    approval_request_id: str = Field(min_length=1, max_length=200)
    principal: str = Field(min_length=1, max_length=200)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["confirm", "reject"]
    note: str = Field(min_length=10, max_length=4000)
    request_id: str = Field(min_length=1, max_length=200)


def _directory_digest(config):
    value = config.model_dump(mode="json")
    if value.get("smtp_security") == "ssl":
        # The old transport was always SSL; preserve pending approvals across this upgrade.
        value.pop("smtp_security")
    return digest(value)


def request_approval(service, candidate_id, config: ApprovalConfig, *, actor, request_id):
    actor = service._actor(actor)
    directory_sha = _directory_digest(config)
    request = {
        "action": "request_expert_approval",
        "candidate_id": candidate_id,
        "directory_sha256": directory_sha,
        "actor": actor,
    }
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        candidate = service.store.get(db, "candidate", candidate_id)
        data = candidate["data"]
        if data["stage"] != "discovered":
            raise ConflictError("Expert requests require a discovered candidate")
        owner = data["candidate"]["owner"]
        contact = config.contacts.get(owner)
        if contact is None:
            raise ValueError("Assigned owner has no configured expert directory entry")
        key = f"{data['candidate']['pattern_id']}@{data['candidate']['pattern_version']}"
        pattern = service.store.get(db, "pattern", key)
        if pattern["data"]["status"] != "active":
            raise ConflictError("Candidate pattern is not active")
        if pattern["data"].get("requires_assessment"):
            assessment = data.get("assessment", {})
            if (
                assessment.get("result") != "applicable"
                or assessment.get("subject_digest") != digest(data["candidate"])
                or assessment.get("pattern_digest") != pattern_digest(pattern["data"])
            ):
                raise ConflictError("Expert request requires current applicable assessment")
        context_sha = service.store.evidence(db, {"candidate": candidate, "pattern": pattern})
        identity = "approval_request_" + digest(
            {"candidate": candidate, "directory": directory_sha, "request_id": request_id}
        )
        # Concurrent or repeated callers cannot spam the same version via fresh idempotency IDs.
        existing = db.execute(
            "SELECT id FROM records WHERE kind='approval_request' "
            "AND json_extract(payload,'$.candidate_id')=? "
            "AND json_extract(payload,'$.expected_version')=? "
            "AND json_extract(payload,'$.directory_sha256')=? "
            "AND json_extract(payload,'$.pattern_digest')=? "
            "AND json_extract(payload,'$.status')='pending' "
            "AND json_extract(payload,'$.expires_at')>? LIMIT 1",
            (
                candidate_id,
                candidate["version"],
                directory_sha,
                pattern_digest(pattern["data"]),
                time.time(),
            ),
        ).fetchone()
        if existing:
            result = service.store.get(db, "approval_request", existing[0])
        else:
            approval = {
                "candidate_id": candidate_id,
                "expected_version": candidate["version"],
                "owner": owner,
                "principal": contact.principal,
                "context_sha256": context_sha,
                "candidate_digest": digest(candidate),
                "pattern_digest": pattern_digest(pattern["data"]),
                "pattern_key": key,
                "directory_sha256": directory_sha,
                "expires_at": time.time() + config.expires_seconds,
                "status": "pending",
                "created_by": actor,
            }
            result = service.store.put(db, "approval_request", identity, approval)
            message = {
                "approval_request_id": identity,
                "target": contact.target,
                "owner": owner,
                "candidate_id": candidate_id,
                "path": data["candidate"]["path"],
                "revision": data["candidate"]["repo_revision"],
                "context_sha256": context_sha,
                "problem": pattern["data"]["pattern"]["problem"],
                "preconditions": pattern["data"]["pattern"]["preconditions"],
                "risks": pattern["data"]["pattern"]["risks"],
                "url": config.gateway_url + "/" + quote(identity, safe=""),
                "notice": "Review evidence and explicitly decide. This is not an approval or proven benefit.",
            }
            service.store.put(
                db,
                "notification",
                identity,
                {
                    "status": "pending",
                    "attempts": 0,
                    "message": message,
                    "directory_sha256": directory_sha,
                },
            )
            service.store.event(
                db, candidate_id, "expert_requested", actor, {"approval_request_id": identity}
            )
        service.store.remember(db, request_id, request, result)
        return result


def _current_request(service, db, identity, principal, config):
    row = service.store.get(db, "approval_request", identity)
    data = row["data"]
    contact = config.contacts.get(data["owner"])
    if not contact or principal != data["principal"] or principal != contact.principal:
        raise PermissionError("Authenticated principal is not this request's assigned expert")
    if _directory_digest(config) != data["directory_sha256"]:
        raise ConflictError("Approval configuration changed; create a new expert request")
    return row


def approval_context(service, config, identity, principal):
    with service.store.transaction() as db:
        row = _current_request(service, db, identity, principal, config)
    return {"request": row, "context": service.store.read_evidence(row["data"]["context_sha256"])}


def apply_decision(service, config: ApprovalConfig, decision: Decision, *, identity_proof):
    """Private gateway entry. Call only after verifying the HTTP signature and principal assertion."""
    request = decision.model_dump(mode="json")
    replay_id = "gateway-decision:" + decision.request_id
    with service.store.transaction() as db:
        row = _current_request(
            service, db, decision.approval_request_id, decision.principal, config
        )
        replay = service.store.replay(db, replay_id, request)
        if replay is not None:
            return replay
        data = row["data"]
        if data["status"] != "pending" or data["expires_at"] < time.time():
            raise ConflictError("Approval request is expired or already decided")
        if decision.context_sha256 != data["context_sha256"]:
            raise ConflictError("Decision must bind the exact expert evidence snapshot")
        candidate = service.store.get(db, "candidate", data["candidate_id"])
        pattern = service.store.get(db, "pattern", data["pattern_key"])
        if (
            digest(candidate) != data["candidate_digest"]
            or pattern_digest(pattern["data"]) != data["pattern_digest"]
        ):
            raise ConflictError("Candidate or pattern changed; request a new review")
        identity = {
            "principal": decision.principal,
            "assurance": "authenticated_gateway",
            "approval_request_id": row["id"],
            "proof": identity_proof,
        }
        result = service.transition(
            data["candidate_id"],
            decision.decision,
            actor=data["owner"],
            expected_version=data["expected_version"],
            request_id="bridge:" + row["id"],
            payload={"note": decision.note},
            _db=db,
            _identity=identity,
        )
        data.update(
            status="decided",
            decision=decision.decision,
            receipt_id=result["data"]["approval_id"],
            decision_sha256=service.store.evidence(db, request),
        )
        service.store.put(db, "approval_request", row["id"], data, row["version"])
        continuation = None
        if decision.decision == "confirm":
            continuation = service.store.put(
                db,
                "continuation",
                row["id"],
                {
                    "candidate_id": data["candidate_id"],
                    "approval_id": data["receipt_id"],
                    "status": "ready_for_workbench",
                    "automatic_execution": False,
                    "entry": f"/evolve-candidate {data['candidate_id']} status",
                },
            )
        answer = {
            "candidate": result,
            "approval_id": data["receipt_id"],
            "continuation": continuation,
        }
        service.store.remember(db, replay_id, request, answer)
        service.store.event(
            db,
            data["candidate_id"],
            "authenticated_expert_decision",
            data["owner"],
            {"approval_request_id": row["id"], "receipt_id": data["receipt_id"]},
        )
        return answer


def signature(key, timestamp, path, body):
    value = b"POST\n" + path.encode("utf-8") + b"\n" + timestamp.encode("ascii") + b"\n" + body
    return hmac.new(key.encode("utf-8"), value, hashlib.sha256).hexdigest()


def authenticate(config, timestamp, supplied_signature, path, body):
    try:
        valid_time = abs(time.time() - int(timestamp)) <= 300
    except (ValueError, TypeError):
        valid_time = False
    if not valid_time or not isinstance(supplied_signature, str):
        raise PermissionError("Invalid gateway signature or timestamp")
    expected = signature(secret(config.signing_key_env, minimum_length=32), timestamp, path, body)
    if not hmac.compare_digest(expected, supplied_signature):
        raise PermissionError("Invalid gateway signature or timestamp")
    return {
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "timestamp": timestamp,
        "scheme": "hmac-sha256-gateway-v1",
    }


def _deliver(config, identity, message):
    if config.webhook_url:
        auth = secret(config.webhook_authorization_env)
        return request_json(
            config.webhook_url,
            method="POST",
            body=message,
            headers={"Idempotency-Key": identity, **({"Authorization": auth} if auth else {})},
        )
    email = EmailMessage()
    email["From"] = config.smtp_sender
    email["To"] = message["target"]
    email["Subject"] = "Evolution 待评审：" + message["path"]
    email["Message-ID"] = f"<{identity}@hmopt.local>"
    lines = [
        "请评审以下优化候选；真实优化收益仍需后续验证。",
        f"责任人：{message['owner']}",
        f"候选 ID：{message['candidate_id']}",
        f"路径：{message['path']}",
        f"源码版本：{message['revision']}",
        "",
        "问题假设：",
        message.get("problem", "请查看完整证据。"),
        "",
        "适用前提：",
        *["- " + value for value in message.get("preconditions", [])],
        "",
        "风险与待核实事项：",
        *["- " + value for value in message.get("risks", [])],
        "",
        "请登录企业审批页面查看冻结证据，并明确确认或拒绝：",
        message["url"],
        "直接回复邮件、打开链接或已读均不构成批准。确认后仍需方案评审、代码评审与验证。",
        "",
        f"审批请求：{identity}",
        f"证据摘要：{message['context_sha256']}",
    ]
    email.set_content("\n".join(lines))
    context = ssl.create_default_context()
    connection = (
        smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=15, context=context)
        if config.smtp_security == "ssl"
        else smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15)
    )
    with connection as smtp:
        if config.smtp_security == "starttls":
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
        if config.smtp_username:
            smtp.login(config.smtp_username, secret(config.smtp_password_env))
        smtp.send_message(email)


def deliver_notifications(service, config, *, actor, limit=20, transport=None):
    """Explicit operator action. Uncertain sends require explicit version-bound requeue."""
    actor = service._actor(actor)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Notification limit must be 1..100")
    transport = transport or _deliver
    results = []
    for _ in range(limit):
        with service.store.transaction() as db:
            found = db.execute(
                "SELECT id FROM records WHERE kind='notification' "
                "AND json_extract(payload,'$.status')='pending' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if not found:
                break
            row = service.store.get(db, "notification", found[0])
            approval = service.store.get(db, "approval_request", found[0])["data"]
            data = row["data"]
            candidate = service.store.get(db, "candidate", approval["candidate_id"])
            pattern = service.store.get(db, "pattern", approval["pattern_key"])
            stale = (
                data["directory_sha256"] != _directory_digest(config)
                or approval["status"] != "pending"
                or approval["expires_at"] < time.time()
                or digest(candidate) != approval["candidate_digest"]
                or pattern["data"]["status"] != "active"
                or pattern_digest(pattern["data"]) != approval["pattern_digest"]
            )
            data.update(
                status="obsolete" if stale else "sending",
                attempts=data["attempts"] + (0 if stale else 1),
            )
            data.pop("error_type", None)
            data.pop("smtp_code", None)
            row = service.store.put(db, "notification", row["id"], data, row["version"])
            if stale:
                service.store.event(db, row["id"], "notification_obsolete", actor, {})
        if stale:
            results.append(row)
            continue
        try:
            transport(config, row["id"], data["message"])
            status = "sent"
        except (OSError, ValueError, smtplib.SMTPException) as error:
            status = "uncertain"
            # Server responses can contain credentials or message content. Keep only safe metadata.
            data["error_type"] = type(error).__name__
            if isinstance(error, smtplib.SMTPResponseException):
                data["smtp_code"] = error.smtp_code
        with service.store.transaction() as db:
            data["status"] = status
            row = service.store.put(db, "notification", row["id"], data, row["version"])
            service.store.event(
                db,
                row["id"],
                "notification_" + status,
                actor,
                {key: data[key] for key in ("attempts", "error_type", "smtp_code") if key in data},
            )
        results.append(row)
    return {"notifications": results, "delivery_semantics": "uncertain sends are not auto-retried"}


def retry_notification(service, identity, *, actor, expected_version, delivery_stopped=False):
    actor = service._actor(actor)
    with service.store.transaction() as db:
        row = service.store.get(db, "notification", identity)
        if row["data"]["status"] == "sending" and not delivery_stopped:
            raise ConflictError(
                "Confirm the previous delivery process stopped before requeueing sending"
            )
        if row["version"] != expected_version or row["data"]["status"] not in {
            "uncertain",
            "sending",
        }:
            raise ConflictError("Only the exact uncertain notification can be requeued")
        data = {**row["data"], "status": "pending"}
        result = service.store.put(db, "notification", identity, data, expected_version)
        service.store.event(
            db,
            identity,
            "notification_retry_authorized",
            actor,
            {"duplicate_delivery_possible": True},
        )
        return result
