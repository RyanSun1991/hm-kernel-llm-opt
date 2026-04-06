# Flash Device Operations

This skill defines the flash protocol that agents must follow when flashing kernel images to the target device via the Flash MCP and Windows relay.

## Architecture

```
Server PC (Flash MCP, port 7337)  →  Windows PC (Relay, port 9100)  →  Phone (USB)
```

The Flash MCP on the server sends commands to a lightweight relay service on the Windows PC. The relay executes fastboot/hdc commands locally where the phone is physically connected via USB.

## Prerequisites

Before any flash operation, the tester agent MUST:

1. Call `relay_health` to verify the Windows relay is reachable.
2. Call `list_devices` to confirm the target device is visible to fastboot.
3. Verify the image file exists and is accessible from the Windows side (shared drive or pre-transferred).

If any prerequisite fails, report the failure immediately. Do NOT proceed with flash.

## Flash Sequence

### Single Partition Flash

Use `flash_device` for flashing a single partition:

```
flash_device(partition="boot", image_path="/path/to/boot.img", device_serial="<serial>")
```

### Flash + Reboot + Wait (Recommended)

Use `flash_and_boot` for the full orchestrated sequence:

```
flash_and_boot(partition="boot", image_path="/path/to/boot.img", device_serial="<serial>")
```

This performs:
1. `fastboot flash <partition> <image>` via relay
2. `fastboot reboot` via relay
3. Poll `hdc list targets` until device appears

### Async Flash

For long-running flash operations, use `flash_and_boot_async` which returns a `task_id` immediately. Poll with `flash_status(task_id=...)`.

## Image Path Translation

Images are built on the Server PC but must be accessible from the Windows PC. Two approaches:

1. **Shared network drive (recommended)**: Map server build output to a Windows drive letter (e.g., `Z:\builds`). Configure `HMOPT_FLASH_SERVER_IMAGE_PREFIX` and `HMOPT_FLASH_WINDOWS_IMAGE_PREFIX` for automatic path translation.
2. **Direct Windows path**: If the image is already on the Windows PC, pass the Windows path directly.

## Error Recovery

- **Flash failure**: Report the exact fastboot error. Do NOT retry automatically — flash failures may indicate device state issues.
- **Reboot failure**: Report and suggest checking USB connection.
- **Boot timeout**: Report elapsed time and suggest checking device state manually. The device may be in a boot loop.
- **Relay unreachable**: The Flash MCP retries 3 times with exponential backoff. If still unreachable, report as infrastructure failure.

## Stock vs Feature Images

In A/B testing workflows:

- **Stock image**: The baseline kernel image without patches. Located at `HMOPT_FLASH_STOCK_IMAGE_DIR` or built from the clean branch.
- **Feature image**: The kernel image with the optimization patch applied. Produced by Build MCP.

The tester agent must flash and test both images in sequence. See `ab-test-comparison.md` for the full A/B comparison protocol.

## Security

The relay enforces a command allowlist (`fastboot`, `hdc`, `adb`, `ping` only). An optional shared secret (`X-Relay-Secret` header) prevents unauthorized access. Always configure `RELAY_SECRET` and `HMOPT_FLASH_RELAY_SECRET` in production.
