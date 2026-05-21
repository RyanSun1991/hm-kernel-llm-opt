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


# ---------------------------------------------------------------------------
# sanity_fix_entries — the asymmetry between Yocto's `git/` (mirrors source
# tree) and `build/` (often a flattened workdir without the wrapping
# subdir) means `--map recipe=path` cannot encode both correctly with a
# single replacement target. Post-parse, we fall back to dirname(file)
# whenever `directory` doesn't exist on disk.
# ---------------------------------------------------------------------------


def test_sanity_fix_replaces_missing_directory_with_dirname_of_file(tmp_path):
    real_src_dir = tmp_path / "kernel" / "sysmgr" / "activation"
    real_src_dir.mkdir(parents=True)
    src = real_src_dir / "actv_bind.c"
    src.write_text("int actv_bind(void){return 0;}\n")

    # `directory` is missing `sysmgr/` — exactly the failure mode the
    # HarmonyOS verif-kernel compdb exhibits.
    entries = [
        {
            "directory": str(tmp_path / "kernel" / "activation"),
            "file": str(src),
            "command": "clang -c actv_bind.c",
        }
    ]
    fixed, dropped = module.sanity_fix_entries(entries)
    assert fixed == 1
    assert dropped == 0
    assert entries[0]["directory"] == str(real_src_dir)
    # other fields untouched
    assert entries[0]["file"] == str(src)
    assert entries[0]["command"] == "clang -c actv_bind.c"


def test_sanity_fix_leaves_valid_entries_alone(tmp_path):
    real_dir = tmp_path / "kernel" / "sysmgr"
    real_dir.mkdir(parents=True)
    src = real_dir / "main.c"
    src.write_text("int main(void){return 0;}\n")

    entries = [
        {
            "directory": str(real_dir),
            "file": str(src),
            "command": "clang -c main.c",
        }
    ]
    fixed, _ = module.sanity_fix_entries(entries)
    assert fixed == 0
    assert entries[0]["directory"] == str(real_dir)


def test_sanity_fix_skips_relative_file_paths(tmp_path):
    """When `file` is relative, dirname(file) is meaningless without joining
    against `directory`, and `directory` is by hypothesis broken — so we
    leave the entry alone rather than synthesize a guess."""
    entries = [
        {
            "directory": str(tmp_path / "nonexistent"),
            "file": "actv_bind.c",
            "command": "clang -c actv_bind.c",
        }
    ]
    fixed, _ = module.sanity_fix_entries(entries)
    assert fixed == 0
    # entry remains unchanged
    assert entries[0]["directory"].endswith("nonexistent")


def test_sanity_fix_skips_when_neither_directory_nor_file_parent_exists(tmp_path):
    """If both `directory` and `dirname(file)` are missing we can't fix
    anything; leave the entry alone so the downstream consumer can surface
    a clear error rather than silently swap one broken path for another."""
    entries = [
        {
            "directory": str(tmp_path / "ghost-build"),
            "file": str(tmp_path / "ghost-src" / "a.c"),
            "command": "clang -c a.c",
        }
    ]
    fixed, _ = module.sanity_fix_entries(entries)
    assert fixed == 0


def test_sanity_fix_caches_directory_existence_check(tmp_path, monkeypatch):
    """Many entries share the same `directory` field; we should not stat the
    same path N times."""
    real_dir = tmp_path / "kernel" / "sysmgr"
    real_dir.mkdir(parents=True)
    src1 = real_dir / "a.c"
    src2 = real_dir / "b.c"
    src1.write_text("")
    src2.write_text("")

    broken = str(tmp_path / "kernel" / "missing")
    entries = [
        {"directory": broken, "file": str(src1), "command": "clang -c a.c"},
        {"directory": broken, "file": str(src2), "command": "clang -c b.c"},
        {"directory": str(real_dir), "file": str(src1), "command": "clang -c a.c"},
    ]

    real_isdir = module.os.path.isdir
    call_log: list = []

    def counting_isdir(p):
        call_log.append(p)
        return real_isdir(p)

    monkeypatch.setattr(module.os.path, "isdir", counting_isdir)

    fixed, _ = module.sanity_fix_entries(entries)

    assert fixed == 2  # both broken entries were rewritten
    # cache: each unique path should be statted at most once. We see:
    #   broken (1st entry) -> miss, stat; broken (2nd entry) -> cache hit;
    #   real_dir (1st fixup target) -> stat; real_dir (3rd entry directory)
    #   -> cache hit.
    assert call_log.count(broken) == 1
    assert call_log.count(str(real_dir)) == 1
