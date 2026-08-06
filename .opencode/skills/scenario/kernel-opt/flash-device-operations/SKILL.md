---
name: flash-device-operations
description: Flash protocol for deploying stock and feature kernel images to the target device via Flash MCP and Windows relay, including the mandatory 10-minute post-flash settle window.
---

# Flash Device Operations

This skill defines the flash protocol that agents must follow when flashing kernel images to the target device via the Flash MCP and Windows relay.

## Architecture

```
Build Server (images) --pscp--> Windows PC (Relay:9100) --USB--> Phone
                                     ^
Server PC (Flash MCP:7337) ----------|
```

- Build Server produces signed images (e.g. `/home/damon/ws/hm/hm-ci-signing/master/nsv_user/`)
- Flash MCP on the Server PC orchestrates the full sequence via the relay
- Windows PC runs the relay service, executes fastboot/hdc/pscp commands locally
- Phone is connected to the Windows PC via USB

## Prerequisites

Before any flash operation, the tester agent MUST:

1. Call `relay_health` to verify the Windows relay is reachable.
2. Call `list_hdc_targets` to confirm the device is connected via hdc.
3. Confirm SCP credentials are configured (HMOPT_FLASH_SCP_HOST, HMOPT_FLASH_SCP_USER).

If any prerequisite fails, report the failure immediately. Do NOT proceed with flash.

## Full Flash Sequence (Integrated Pipeline)

The `flash_and_boot` tool sends a **single command** to the relay that runs `flash_pipeline.py` on the Windows PC. The entire sequence executes locally on Windows with maximum determinism — no multi-step HTTP round-trips:

```
flash_pipeline.py on Windows:
  1. pscp images from build server → local dir
  2. hdc shell reboot bootloader
  3. wait for device in fastboot (poll)
  4. fastboot flash boot boot_plr.img
     fastboot flash modem_driver modem_driver_plr.img
  5. fastboot reboot
  6. wait for device in hdc list targets (poll)
```

The script outputs a single JSON result to stdout with success/failure and step details.

### Why Integrated Script?

- **Determinism**: Full sequence runs as one local process on Windows, no HTTP round-trips between steps
- **Reliability**: No relay connection drops between steps
- **Speed**: No LLM decision overhead between steps
- **Atomicity**: Single success/failure result for the entire pipeline

### Example: Flash Stock Image (A/B Baseline)

The simplest way — uses `HMOPT_FLASH_STOCK_IMAGE_DIR` and `HMOPT_FLASH_DEFAULT_PARTITIONS` from env:

```
flash_stock(device_serial="<serial>")
```

### Example: Flash Feature Image (A/B Candidate)

Uses `HMOPT_FLASH_FEATURE_IMAGE_DIR` and `HMOPT_FLASH_DEFAULT_PARTITIONS` from env:

```
flash_feature(device_serial="<serial>")
```

### Example: Flash with Explicit Image Transfer

For custom paths not matching the env defaults:

```
flash_and_boot(
    server_images=[
        {"server_path": "/home/damon/ws/hm/signing/nsv_user/boot.img", "local_filename": "boot_plr.img"},
        {"server_path": "/home/damon/ws/hm/signing/nsv_user/modem_driver.img", "local_filename": "modem_driver_plr.img"},
    ],
    partitions=[
        {"partition": "boot", "image_path": "boot_plr.img"},
        {"partition": "modem_driver", "image_path": "modem_driver_plr.img"},
    ],
    device_serial="<serial>",
)
```

### Example: Flash with Pre-transferred Images

If images are already on the Windows PC, omit `server_images`:

```
flash_and_boot(
    partitions=[
        {"partition": "boot", "image_path": "boot_plr.img"},
        {"partition": "modem_driver", "image_path": "modem_driver_plr.img"},
    ],
)
```

## Individual Tools

| Tool | Purpose |
|---|---|
| **`flash_stock`** | **Flash stock (baseline) image — auto-resolves paths from `HMOPT_FLASH_STOCK_IMAGE_DIR`** |
| **`flash_feature`** | **Flash feature (patched) image — auto-resolves paths from `HMOPT_FLASH_FEATURE_IMAGE_DIR`** |
| `flash_stock_async` | Async version of `flash_stock`, returns task_id |
| `flash_feature_async` | Async version of `flash_feature`, returns task_id |
| `transfer_images` | Pull images from build server to Windows via pscp |
| `enter_bootloader` | `hdc shell reboot bootloader` to enter fastboot mode |
| `wait_for_fastboot` | Poll `fastboot devices` until device appears |
| `flash_device` | Flash a single partition via fastboot |
| `flash_partitions` | Flash multiple partitions sequentially |
| `device_reboot` | `fastboot reboot` |
| `device_wait_boot` | Poll `hdc list targets` until device appears |
| `flash_and_boot` | Full orchestrated pipeline (all steps above) |
| `flash_and_boot_async` | Async version, returns task_id |
| `flash_status` | Query async task status |
| `relay_health` | Check relay connectivity |
| `list_devices` | List fastboot-visible devices |
| `list_hdc_targets` | List hdc-visible devices |

## Image Transfer via pscp

Images are pulled from the build server to the Windows PC using `pscp` (PuTTY SCP):

```
pscp -pw <password> <user>@<host>:<server_path> <local_filename>
```

Configure via environment variables:
- `HMOPT_FLASH_SCP_TOOL` — `pscp` (default) or `scp`
- `HMOPT_FLASH_SCP_HOST` — Build server IP (e.g. `10.123.104.91`)
- `HMOPT_FLASH_SCP_USER` — SSH username (e.g. `damon`)
- `HMOPT_FLASH_SCP_PASSWORD` — SSH password (for pscp `-pw` flag)
- `HMOPT_FLASH_WINDOWS_IMAGE_DIR` — Local directory on Windows for images (default: `.`)

## Post-Flash Settle Window — Mandatory

`flash_and_boot` (and therefore `flash_stock` / `flash_feature`) only waits for the device to reappear in `hdc list targets` — that confirms the kernel booted, but userspace (xdevice agents, perf counters, UI services, settings app) takes several more minutes to settle.  Kicking off a test immediately after flash returns produces:

- xdevice "device not supported" / connection-not-ready warnings
- missing or truncated reports
- flaky A/B deltas caused by settle-time overhead, not the patch

**After every successful flash, wait ~10 minutes (600 s) before running a test or any downstream protocol that depends on a ready device.**  Use Bash `sleep 600` (or equivalent).  Apply to **both** stock and feature independently — never parallelize.

During the settle, optionally poll `list_hdc_targets()` every 60 s as a liveness check.  If hdc loses the device during the window, the flash didn't land cleanly — mark the phase **skipped** and do not proceed.

## Error Recovery

- **Transfer failure**: Report the exact pscp error. Check network connectivity and credentials.
- **Bootloader entry failure**: Device may not be in a state where hdc works. Check USB connection.
- **Fastboot wait timeout**: Device may not have entered bootloader. Try `hdc shell reboot bootloader` again.
- **Flash failure**: Report the exact fastboot error. Do NOT retry automatically.
- **Boot timeout**: Device may be in a boot loop. Report elapsed time and suggest manual check.
- **Relay unreachable**: Flash MCP retries 3 times with exponential backoff.

## Stock vs Feature Images

In A/B testing workflows:

- **Stock image**: Baseline kernel without patches. Path configured via `HMOPT_FLASH_STOCK_IMAGE_DIR`.
- **Feature image**: Kernel with the optimization patch. Path configured via `HMOPT_FLASH_FEATURE_IMAGE_DIR`.
- **Default partitions**: Configured via `HMOPT_FLASH_DEFAULT_PARTITIONS` (default: `boot:boot.img,modem_driver:modem_driver.img`).

The tester agent uses the simplified `flash_stock` / `flash_feature` tools:

```
# Phase A: Stock baseline
flash_stock(device_serial="<serial>")
# ... run stock test ...

# Phase B: Feature candidate
flash_feature(device_serial="<serial>")
# ... run feature test ...
```

Both tools auto-resolve `server_images` from the configured image directory and default partitions. Override via arguments when needed (e.g., when Build MCP outputs images to a non-default path).

See `ab-test-comparison/SKILL.md` for the full A/B protocol.

## Security

The relay enforces a command allowlist (`fastboot`, `hdc`, `adb`, `pscp`, `scp`, `ping` only). An optional shared secret (`X-Relay-Secret` header) prevents unauthorized access. Always configure `RELAY_SECRET` and `HMOPT_FLASH_RELAY_SECRET` in production.
