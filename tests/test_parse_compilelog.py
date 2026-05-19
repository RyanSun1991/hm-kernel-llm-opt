import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "parse_compilelog", Path(__file__).resolve().parents[1] / "scripts" / "parse_compilelog.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_build_transform_maps_git_and_build_paths():
    transform = module.build_transform(
        "/home/ryan/code/scratch/tongkun/hione/",
        "/work/trunk",
        {
            "hm-sysmgr-nashvilleoh": "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel"
        },
    )

    git_path = (
        "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-nashvilleoh/git-r0/git/sysmgr/main.c"
    )
    build_path = (
        "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-nashvilleoh/git-r0/build/generated/hmsd/hook/sysmgr/time.c"
    )

    assert transform(git_path).endswith("/hm-verif-kernel/sysmgr/main.c")
    assert transform(build_path).endswith("/hm-verif-kernel/generated/hmsd/hook/sysmgr/time.c")


def test_build_transform_maps_docker_to_host_trunk_first():
    transform = module.build_transform(
        "/home/ryan/code/scratch/tongkun/hione/",
        "/work/trunk",
        {},
    )
    path = "/work/trunk/kernel/hongmeng/hm-verif-kernel/sysmgr/include/a.h"
    assert transform(path) == "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel/sysmgr/include/a.h"


def test_build_transform_map_works_without_host_prefix_or_docker_trunk_match():
    transform = module.build_transform(
        None,
        "/work/trunk_new",
        {
            "hm-sysmgr-nashvilleoh": "/home/aoxiang/hione/PLA-phone/trunk/kernel/hongmeng/hm-verif-kernel"
        },
    )

    host_git_path = (
        "/home/aoxiang/hione/PLA-phone/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-nashvilleoh/git-r0/git/sysmgr/exts/hisi/hp_turbo.c"
    )
    host_build_path = (
        "/home/aoxiang/hione/PLA-phone/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-nashvilleoh/git-r0/build/exts/hisi/hisiplatform/basicplatform/swap_dma/hp_turbo"
    )

    assert transform(host_git_path).endswith("/hm-verif-kernel/sysmgr/exts/hisi/hp_turbo.c")
    assert transform(host_build_path).endswith(
        "/hm-verif-kernel/exts/hisi/hisiplatform/basicplatform/swap_dma/hp_turbo"
    )


def test_build_transform_with_source_subdir_keeps_sysmgr_for_build_paths():
    """When a yocto recipe builds from a source subdir (e.g. hm-sysmgr-* builds from
    <repo>/sysmgr/), the build/ tree mirrors the source subdir, not the repo root.
    Without :source_subdir the build/ replacement drops the in-repo prefix, which
    produces compile_commands.json entries whose `directory` is missing one segment
    and causes scip-clang's clang frontend to fail with "unable to set working
    directory: ..." and then SIGSEGV. The :source_subdir suffix fixes this.
    """
    transform = module.build_transform(
        None,
        "/work/trunk",
        {
            "hm-sysmgr-charlotteoh": (
                "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/hm-verif-kernel",
                "sysmgr",
            )
        },
    )

    git_path = (
        "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-charlotteoh/git-r0/git/sysmgr/activation/actv_bind.c"
    )
    build_dir = (
        "/home/ryan/code/scratch/tongkun/hione/work/trunk/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-charlotteoh/git-r0/build/activation"
    )

    assert transform(git_path).endswith("/hm-verif-kernel/sysmgr/activation/actv_bind.c")
    # The fix: build/ paths get an extra /sysmgr/ inserted, so directory entries
    # now resolve to a real on-disk path.
    assert transform(build_dir).endswith("/hm-verif-kernel/sysmgr/activation")


def test_main_parses_map_with_optional_source_subdir(tmp_path, monkeypatch):
    """End-to-end: the --map CLI arg accepts recipe=path:source_subdir."""
    import sys

    log = tmp_path / "log"
    log.write_text(
        "make[1]: Entering directory '/abs/build_tools/yocto/ng/build/tmp/work/"
        "aarch64-euler-elf/hm-sysmgr-charlotteoh/git-r0/build/activation'\n"
        "clang -c /abs/build_tools/yocto/ng/build/tmp/work/aarch64-euler-elf/"
        "hm-sysmgr-charlotteoh/git-r0/git/sysmgr/activation/actv_bind.c -o foo.o\n"
    )
    out = tmp_path / "compile_commands.json"

    argv = [
        "parse_compilelog.py",
        "-i", str(log),
        "-o", str(out),
        "--map",
        "hm-sysmgr-charlotteoh=/repo/hm-verif-kernel:sysmgr",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    import json
    entries = json.loads(out.read_text())
    assert len(entries) == 1
    e = entries[0]
    assert e["directory"].endswith("/hm-verif-kernel/sysmgr/activation")
    assert e["file"].endswith("/hm-verif-kernel/sysmgr/activation/actv_bind.c")
