# Build + Sign Feature Image

This skill defines the Build MCP workflow the tester runs to produce a signed feature image before it can be flashed.  It ends when a ready-to-flash image exists in the directory the Flash MCP reads from.

## Scope

- Build MCP only.  Does NOT flash.  Does NOT run tests.
- Runs on the feature (patched) branch only.  Stock images are prebuilt and pulled by `flash-device-operations.md`.

## Tools

| Tool | Purpose |
|---|---|
| Build MCP `kernel_build_trigger` | Compile the kernel with the current branch's patches |
| Build MCP `kernel_sign_trigger` | Package / sign the built image into a flashable artifact |
| Build MCP `kernel_build_status` (if async) | Poll a build task by id |

## Preconditions

Before invoking Build MCP:

1. The patch under review has been committed to the feature branch (check `git log`).
2. `HMOPT_FLASH_FEATURE_IMAGE_DIR` is configured and points at the **signed** output directory (not the raw build tree).
3. The sign step's output path matches what `flash_feature` expects — if `HMOPT_FLASH_DEFAULT_PARTITIONS` lists `boot:boot.img`, the sign output must contain `boot.img` (or `boot_plr.img` after translation) under that dir.

If any precondition fails: verdict = **fail**, return to manager.

## Mandatory Sequence

```
# Step 1: build the patched kernel
build_result = kernel_build_trigger(
    # branch / config / target per the staged task
)
# Confirm build_result.success is True.  If False:
#   - read build_result.stderr_tail for the compiler error
#   - set verdict=fail, record the error, return to manager
#   - do NOT attempt sign on a failed build

# Step 2: sign + package the built image
sign_result = kernel_sign_trigger(
    # usually no arguments; defaults target the build output
)
# Confirm sign_result.success is True.  If False:
#   - verdict=fail, record sign_result error
#   - return to manager
```

Both steps are synchronous in typical configurations.  If your Build MCP variant is async, submit the task and poll `kernel_build_status(task_id)` with a 60 s cadence.

## Postcondition

After both steps succeed:

- A signed feature image exists under `HMOPT_FLASH_FEATURE_IMAGE_DIR` matching every partition in `HMOPT_FLASH_DEFAULT_PARTITIONS`.
- `flash-device-operations.md`'s `flash_feature` will pick it up without further arguments.

## Failure Modes

| Phase | Failure signal | Verdict | Next action |
|---|---|---|---|
| build | non-zero returncode, compiler error in stderr | fail | report build log tail to manager; do NOT sign |
| sign | non-zero returncode, signing tool error | fail | report sign log tail; feature image unusable |
| postcondition | image dir empty or missing expected partition | fail | investigate build/sign output paths; flag env config |

## Hard Rules

1. NEVER flash without signing — `flash_feature` reads the **signed** output dir, not the raw build tree.  A raw `.img` from the build tree may boot differently or fail secure-boot checks.
2. NEVER skip the sign step "because only kernel code changed".  Sign also packages ancillary partitions the device expects.
3. NEVER retry a failed build automatically.  Compiler failures usually mean the patch doesn't apply cleanly or introduces a syntax/type error — surface it to the manager, let the coder agent fix it.

## Reporting

Hand back to `kernel-tester-agent` (which invoked this skill):

- `build.success`, duration, key stderr lines on failure
- `sign.success`, duration, output directory path
- signed artifact filenames that ended up in `HMOPT_FLASH_FEATURE_IMAGE_DIR` (useful for the validation report)

The tester then proceeds to `flash-device-operations.md`.
