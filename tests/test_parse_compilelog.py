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
