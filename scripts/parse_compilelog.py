#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shlex
import sys
from typing import Dict, List, Optional, Tuple

MAKE_ENTER_RE = re.compile(r"Entering directory ['`](.+?)['`]")
CD_AND_RE = re.compile(r"^\s*cd\s+(\S+)\s*&&\s*(.+)$")
CLANG_HINT_RE = re.compile(r"(?:^|\s)(\S*clang\+\+|\S*clang)(?:\s|$)")


def build_transform(
    host_prefix: Optional[str],
    docker_trunk: str,
    maps: Dict[str, str],
):
    """
    Transform path fragments in compile commands.

    1) docker trunk -> host trunk:
       /work/trunk_new/... -> <host_prefix>/work/trunk_new/...
    2) yocto tmp/work recipe workdir -> mapped repo root (supports git-r0/git and git-r0/build):
       .../tmp/work/<triplet>/<recipe>/git-r0/git/...   -> <mapped_path>/...
       .../tmp/work/<triplet>/<recipe>/git-r0/build/... -> <mapped_path>/...

    NOTE:
    - map replacement intentionally does not depend on --docker-trunk/--host-prefix values,
      so it also works for host absolute logs where only --map is provided.
    """
    docker_trunk = docker_trunk.rstrip("/")
    host_trunk = None
    if host_prefix:
        host_prefix = host_prefix.rstrip("/")
        host_trunk = (
            f"{host_prefix}{docker_trunk}"
            if docker_trunk.startswith("/")
            else f"{host_prefix}/{docker_trunk}"
        )

    yocto_rules: List[Tuple[re.Pattern, str]] = []
    for recipe, target in maps.items():
        target = target.rstrip("/") + "/"
        # match both host and docker style prefixes by anchoring from tmp/work onward
        pat = re.compile(
            r"(?:^|/)"
            r"[^\s]*?/build_tools/yocto/ng/build/tmp/work/"
            r"[^/]+/"
            + re.escape(recipe)
            + r"/git-r0/(?:git|build)/"
        )
        yocto_rules.append((pat, target))

    def transform(text: str) -> str:
        out = text

        # 1) docker trunk -> host trunk
        if host_trunk and host_prefix:
            out = out.replace(host_trunk + "/", "__HOSTTRUNK__/")
            out = out.replace(docker_trunk + "/", host_trunk + "/")
            out = out.replace("__HOSTTRUNK__/", host_trunk + "/")
            out = out.replace(host_prefix + host_prefix, host_prefix)

        # 2) yocto recipe workdir -> mapped repo root
        for pat, repl in yocto_rules:
            out = pat.sub(repl, out)

        return out

    return transform


def extract_file_from_cmd(cmd: str) -> Optional[str]:
    m = re.search(r"\s-c\s+(\S+)", cmd)
    if not m:
        return None

    try:
        argv = shlex.split(cmd, posix=True)
    except Exception:
        return m.group(1)

    for i, tok in enumerate(argv):
        if tok == "-c" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("-c") and len(tok) > 2:
            return tok[2:]
    return m.group(1)


def is_compile_line(s: str) -> bool:
    if not CLANG_HINT_RE.search(s):
        return False
    if " llvm-ar " in s or " llvm-ranlib " in s:
        return False
    return (" -c " in s) or bool(re.search(r"\s-c\s+\S+", s))


def parse_log(stream, transform) -> List[dict]:
    entries: List[dict] = []
    seen = set()

    current_dir: Optional[str] = None
    pending: Optional[str] = None

    for raw in stream:
        line = raw.rstrip("\n")

        m_enter = MAKE_ENTER_RE.search(line)
        if m_enter:
            current_dir = m_enter.group(1)

        if pending is not None:
            pending += " " + line.strip()
            if is_compile_line(pending):
                line = pending
                pending = None
            else:
                continue

        if CLANG_HINT_RE.search(line) and (not is_compile_line(line)):
            pending = line.strip()
            continue

        if not is_compile_line(line):
            continue

        directory = current_dir or "."
        command = line.strip()

        m_cd = CD_AND_RE.match(command)
        if m_cd:
            directory = m_cd.group(1)
            command = m_cd.group(2).strip()

        src = extract_file_from_cmd(command)
        if not src:
            continue

        if not os.path.isabs(src) and directory not in (None, ".", ""):
            src = os.path.normpath(os.path.join(directory, src))

        directory_t = transform(directory)
        command_t = transform(command)
        file_t = transform(src)

        key = (directory_t, file_t, command_t)
        if key in seen:
            continue
        seen.add(key)

        entries.append({
            "directory": directory_t,
            "command": command_t,
            "file": file_t,
        })

    return entries


def main():
    ap = argparse.ArgumentParser(description="Parse Yocto build log -> compile_commands.json")
    ap.add_argument("-i", "--input", default="-", help="build log path, or '-' for stdin")
    ap.add_argument("-o", "--output", default="compile_commands.json", help="output compile_commands.json")
    ap.add_argument(
        "--host-prefix",
        default="",
        help="e.g. /home/ryan/code/scratch/tongkun/hione ; if empty, no /work->host mapping",
    )
    ap.add_argument("--docker-trunk", default="/work/trunk_new", help="docker trunk prefix")
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        help=(
            "recipe=real_repo_root ; replaces "
            ".../<recipe>/git-r0/git/ and .../<recipe>/git-r0/build/ -> real_repo_root/"
        ),
    )
    args = ap.parse_args()

    maps: Dict[str, str] = {}
    for item in args.map:
        if "=" not in item:
            ap.error(f"--map must be like recipe=/abs/path , got: {item}")
        k, v = item.split("=", 1)
        maps[k.strip()] = v.strip()

    host_prefix = args.host_prefix.strip() or None
    transform = build_transform(host_prefix, args.docker_trunk, maps)

    if args.input in ("-", ""):
        stream = sys.stdin
    else:
        stream = open(args.input, "r", errors="replace")

    try:
        entries = parse_log(stream, transform)
    finally:
        if stream is not sys.stdin:
            stream.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entries)} entries to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
