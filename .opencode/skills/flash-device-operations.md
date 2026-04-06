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

## Full Flash Sequence (Production Workflow)

The `flash_and_boot` tool executes the complete pipeline:

```
1. pscp images from build server → Windows PC        (transfer_images)
2. hdc shell reboot bootloader                        (enter_bootloader)
3. wait ~15s for device to enter bootloader
4. poll fastboot devices until device appears          (wait_for_fastboot)
5. fastboot flash boot boot_plr.img                    (flash_partitions)
   fastboot flash modem_driver modem_driver_plr.img
6. wait 2s
7. fastboot reboot                                     (device_reboot)
8. poll hdc list targets until device appears           (device_wait_boot)
```

### Example: Flash with Image Transfer

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

## Error Recovery

- **Transfer failure**: Report the exact pscp error. Check network connectivity and credentials.
- **Bootloader entry failure**: Device may not be in a state where hdc works. Check USB connection.
- **Fastboot wait timeout**: Device may not have entered bootloader. Try `hdc shell reboot bootloader` again.
- **Flash failure**: Report the exact fastboot error. Do NOT retry automatically.
- **Boot timeout**: Device may be in a boot loop. Report elapsed time and suggest manual check.
- **Relay unreachable**: Flash MCP retries 3 times with exponential backoff.

## Stock vs Feature Images

In A/B testing workflows:

- **Stock image**: Baseline kernel without patches. Built from the clean branch, signed images at the stock signing path.
- **Feature image**: Kernel with the optimization patch. Built and signed via Build MCP.

The tester agent must flash and test both in sequence. See `ab-test-comparison.md` for the full A/B protocol.

## Security

The relay enforces a command allowlist (`fastboot`, `hdc`, `adb`, `pscp`, `scp`, `ping` only). An optional shared secret (`X-Relay-Secret` header) prevents unauthorized access. Always configure `RELAY_SECRET` and `HMOPT_FLASH_RELAY_SECRET` in production.
