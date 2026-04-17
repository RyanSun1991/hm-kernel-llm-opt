"""All-in-one flash pipeline script for Windows.

Runs the complete flash sequence locally on the Windows PC:
  1. pscp images from build server
  2. hdc shell reboot bootloader
  3. wait for fastboot device
  4. fastboot flash (multiple partitions)
  5. fastboot reboot
  6. wait for hdc device

Zero external dependencies — uses only Python stdlib.
Outputs a single JSON result to stdout for machine consumption.

Usage:
  python flash_pipeline.py \
    --scp-host 10.123.104.91 --scp-user damon --scp-pw 1 \
    --image /home/damon/ws/signing/nsv_user/boot.img:boot:boot_plr.img \
    --image /home/damon/ws/signing/nsv_user/modem_driver.img:modem_driver:modem_driver_plr.img \
    --image-dir C:\images \
    --bootloader-wait 10 \
    --boot-wait-timeout 300

  Each --image is: <server_path>:<partition>:<local_filename>
  If server_path is empty (e.g. ":boot:boot_plr.img"), skips SCP transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any


def _run(argv: list[str], timeout_s: int = 120, cwd: str | None = None) -> dict[str, Any]:
    """Run a subprocess and return structured result."""
    cmd_str = " ".join(argv)
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": round(time.time() - started, 3),
            "command": cmd_str,
        }
    except FileNotFoundError:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"not found: {argv[0]}", "duration_s": round(time.time() - started, 3), "command": cmd_str}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -9, "stdout": "", "stderr": f"timeout after {timeout_s}s", "duration_s": round(time.time() - started, 3), "command": cmd_str}


def _log(msg: str) -> None:
    print(f"[flash] {msg}", file=sys.stderr, flush=True)


def run_pipeline(
    *,
    images: list[dict[str, str]],
    scp_host: str = "",
    scp_user: str = "",
    scp_password: str = "",
    scp_tool: str = "pscp",
    image_dir: str = ".",
    hdc_bin: str = "hdc",
    fastboot_bin: str = "fastboot",
    device_serial: str = "",
    bootloader_wait_s: int = 10,
    fastboot_wait_s: int = 30,
    flash_timeout_s: int = 600,
    boot_wait_timeout_s: int = 300,
    boot_poll_interval_s: int = 5,
    scp_timeout_s: int = 300,
) -> dict[str, Any]:
    """Execute the complete flash pipeline.

    Args:
        images: List of dicts with keys:
            - server_path: remote path on build server (empty to skip transfer)
            - partition: fastboot partition name (e.g. "boot", "modem_driver")
            - local_filename: local filename on Windows
        scp_host/scp_user/scp_password: SCP credentials
        scp_tool: "pscp" or "scp"
        image_dir: working directory for images on Windows
        hdc_bin/fastboot_bin: tool paths
        device_serial: target device serial (optional)
        bootloader_wait_s: seconds to wait after entering bootloader
        fastboot_wait_s: max seconds to wait for fastboot device
        flash_timeout_s: timeout per partition flash
        boot_wait_timeout_s: max seconds to wait for hdc after reboot
        boot_poll_interval_s: poll interval for hdc wait

    Returns:
        JSON-serializable dict with success, phase, and step details.
    """
    steps: dict[str, Any] = {}
    hdc_target = ["-t", device_serial] if device_serial else []
    fb_serial = ["-s", device_serial] if device_serial else []

    # ── Step 1: Transfer images via pscp/scp ──────────────────────────────
    needs_transfer = any(img.get("server_path") for img in images)
    if needs_transfer:
        if not scp_host or not scp_user:
            return {"success": False, "phase": "transfer", "error": "scp_host and scp_user required", "steps": steps}

        _log(f"Transferring {len(images)} images from {scp_host}...")
        transfer_results = []
        for img in images:
            server_path = img.get("server_path", "")
            local_filename = img["local_filename"]
            if not server_path:
                transfer_results.append({"file": local_filename, "skipped": True, "ok": True})
                continue

            remote_spec = f"{scp_user}@{scp_host}:{server_path}"
            local_path = os.path.join(image_dir, local_filename)

            scp_args = [scp_tool]
            if scp_password and scp_tool in ("pscp", "pscp.exe"):
                scp_args.extend(["-pw", scp_password])
            scp_args.extend([remote_spec, local_path])

            r = _run(scp_args, timeout_s=scp_timeout_s)
            transfer_results.append({"file": local_filename, "skipped": False, **r})
            if not r["ok"]:
                steps["transfer"] = transfer_results
                return {"success": False, "phase": "transfer", "error": f"failed to transfer {local_filename}", "steps": steps}
            _log(f"  transferred {local_filename}")

        steps["transfer"] = transfer_results

    # ── Step 2: Enter bootloader via hdc ──────────────────────────────────
    _log("Entering bootloader mode...")
    bl_result = _run([hdc_bin] + hdc_target + ["shell", "reboot", "bootloader"], timeout_s=30)
    steps["enter_bootloader"] = bl_result
    if not bl_result["ok"]:
        return {"success": False, "phase": "enter_bootloader", "steps": steps}

    # ── Step 3: Wait for fastboot device ──────────────────────────────────
    _log(f"Waiting {bootloader_wait_s}s for bootloader...")
    time.sleep(bootloader_wait_s)

    _log("Polling fastboot devices...")
    fb_found = False
    fb_start = time.time()
    while time.time() - fb_start < fastboot_wait_s:
        r = _run([fastboot_bin, "devices"], timeout_s=10)
        stdout = r.get("stdout", "")
        if device_serial:
            if device_serial in stdout:
                fb_found = True
                break
        else:
            lines = [l for l in stdout.strip().splitlines() if l.strip()]
            if lines:
                fb_found = True
                break
        time.sleep(2)

    steps["wait_fastboot"] = {"ok": fb_found, "elapsed_s": round(time.time() - fb_start, 1)}
    if not fb_found:
        return {"success": False, "phase": "wait_fastboot", "error": "device not found in fastboot", "steps": steps}
    _log("Device found in fastboot mode")

    # ── Step 4: Flash partitions ──────────────────────────────────────────
    _log("Flashing partitions...")
    flash_results = []
    for img in images:
        partition = img["partition"]
        local_filename = img["local_filename"]
        local_path = os.path.join(image_dir, local_filename)

        flash_args = [fastboot_bin] + fb_serial + ["flash", partition, local_path]
        r = _run(flash_args, timeout_s=flash_timeout_s)
        flash_results.append({"partition": partition, "file": local_filename, **r})
        if not r["ok"]:
            steps["flash"] = flash_results
            return {"success": False, "phase": "flash", "error": f"failed to flash {partition}", "steps": steps}
        _log(f"  flashed {partition}")

    steps["flash"] = flash_results

    # ── Step 5: Reboot ────────────────────────────────────────────────────
    _log("Rebooting device...")
    time.sleep(2)
    reboot_result = _run([fastboot_bin] + fb_serial + ["reboot"], timeout_s=60)
    steps["reboot"] = reboot_result
    if not reboot_result["ok"]:
        return {"success": False, "phase": "reboot", "steps": steps}

    # ── Step 6: Wait for hdc ──────────────────────────────────────────────
    _log("Waiting for device to boot (hdc)...")
    hdc_found = False
    hdc_start = time.time()
    while time.time() - hdc_start < boot_wait_timeout_s:
        r = _run([hdc_bin, "list", "targets"], timeout_s=10)
        stdout = r.get("stdout", "")
        if device_serial:
            if device_serial in stdout:
                hdc_found = True
                break
        else:
            lines = [l.strip() for l in stdout.strip().splitlines() if l.strip() and l.strip() != "[Empty]"]
            if lines:
                hdc_found = True
                break
        time.sleep(boot_poll_interval_s)

    elapsed_boot = round(time.time() - hdc_start, 1)
    steps["boot_wait"] = {"ok": hdc_found, "elapsed_s": elapsed_boot}
    if not hdc_found:
        return {"success": False, "phase": "boot_wait", "error": f"device not online after {elapsed_boot}s", "steps": steps}

    _log(f"Device online after {elapsed_boot}s. Flash pipeline complete.")
    return {"success": True, "phase": "complete", "steps": steps}


def _parse_image_spec(spec: str) -> dict[str, str]:
    """Parse 'server_path:partition:local_filename' spec."""
    parts = spec.split(":", 2)
    if len(parts) == 3:
        return {"server_path": parts[0], "partition": parts[1], "local_filename": parts[2]}
    if len(parts) == 2:
        return {"server_path": "", "partition": parts[0], "local_filename": parts[1]}
    raise ValueError(f"invalid image spec: {spec!r}; expected server_path:partition:local_filename")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HMOpt Flash Pipeline — all-in-one flash sequence")
    parser.add_argument("--image", action="append", default=[], help="Image spec: server_path:partition:local_filename (repeatable)")
    parser.add_argument("--image-dir", default=os.environ.get("HMOPT_FLASH_WINDOWS_IMAGE_DIR", "."), help="Local image directory")
    parser.add_argument("--scp-host", default=os.environ.get("HMOPT_FLASH_SCP_HOST", ""), help="Build server host")
    parser.add_argument("--scp-user", default=os.environ.get("HMOPT_FLASH_SCP_USER", ""), help="SCP username")
    parser.add_argument("--scp-pw", default=os.environ.get("HMOPT_FLASH_SCP_PASSWORD", ""), help="SCP password (for pscp -pw)")
    parser.add_argument("--scp-tool", default=os.environ.get("HMOPT_FLASH_SCP_TOOL", "pscp"), help="SCP tool: pscp or scp")
    parser.add_argument("--hdc", default="hdc", help="hdc binary path")
    parser.add_argument("--fastboot", default="fastboot", help="fastboot binary path")
    parser.add_argument("--serial", default=os.environ.get("HMOPT_FLASH_DEVICE_SERIAL", ""), help="Device serial")
    parser.add_argument("--bootloader-wait", type=int, default=10, help="Seconds to wait after entering bootloader")
    parser.add_argument("--fastboot-wait", type=int, default=30, help="Max seconds to wait for fastboot device")
    parser.add_argument("--flash-timeout", type=int, default=600, help="Timeout per partition flash")
    parser.add_argument("--boot-wait-timeout", type=int, default=300, help="Max seconds to wait for hdc after reboot")
    parser.add_argument("--boot-poll-interval", type=int, default=5, help="Poll interval for hdc wait")
    parser.add_argument("--scp-timeout", type=int, default=300, help="Timeout per SCP transfer")

    args = parser.parse_args(argv)

    if not args.image:
        print(json.dumps({"success": False, "error": "at least one --image is required"}))
        return 1

    images = [_parse_image_spec(spec) for spec in args.image]

    result = run_pipeline(
        images=images,
        scp_host=args.scp_host,
        scp_user=args.scp_user,
        scp_password=args.scp_pw,
        scp_tool=args.scp_tool,
        image_dir=args.image_dir,
        hdc_bin=args.hdc,
        fastboot_bin=args.fastboot,
        device_serial=args.serial,
        bootloader_wait_s=args.bootloader_wait,
        fastboot_wait_s=args.fastboot_wait,
        flash_timeout_s=args.flash_timeout,
        boot_wait_timeout_s=args.boot_wait_timeout,
        boot_poll_interval_s=args.boot_poll_interval,
        scp_timeout_s=args.scp_timeout,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
