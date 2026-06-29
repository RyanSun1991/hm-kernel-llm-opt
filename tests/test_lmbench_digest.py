"""Unit tests for the lmbench xlsx -> compact digest parser.

The parser lives under tools/windows_relay/ (it must run on the bare Windows
relay), so we add that dir to sys.path. Skips if openpyxl is absent.
"""
import json
import sys
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

_RELAY = Path(__file__).resolve().parents[1] / "tools" / "windows_relay"
sys.path.insert(0, str(_RELAY))
import lmbench_digest as ld  # noqa: E402


def _xlsx(path, header, rows, sheet="result"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(header)
    for r in rows:
        ws.append(list(r))
    wb.save(str(path))


_TOTAL_HDR = ["system", "tool", "metric", "command", "average", "variance ",
              "standard_deviation", "Discrete", "value0", "units"]


def test_direction_classification():
    assert ld._direction("bw_mem -P 1 10m rd", "MB/s") == "higher_better"
    assert ld._direction("lat_syscall -P 1 null", "microseconds") == "lower_better"
    assert ld._direction("lat_ctx -P 1 -s 16 8", "") == "lower_better"
    assert ld._direction("bw_mem write", "") == "higher_better"
    assert ld._direction("weird_metric", "") == "unknown"


def test_norm_signs_by_direction():
    # +5% on a higher-better metric is an improvement (+5); on lower-better it's -5.
    assert ld._norm(5.0, "higher_better") == 5.0
    assert ld._norm(5.0, "lower_better") == -5.0
    assert ld._norm(5.0, "unknown") is None


def test_vs_previous_direction_aware(tmp_path):
    cur = tmp_path / "cur.xlsx"
    prev = tmp_path / "prev.xlsx"
    # bandwidth went up (good); latency went up (bad)
    _xlsx(cur, _TOTAL_HDR, [
        ["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 110.0, 1, 1, 0.01, 110, "MB/s"],
        ["大核", "lmbench-lat", "lat_sys", "lat_syscall null", 22.0, 1, 1, 0.02, 22, "microseconds"],
    ])
    _xlsx(prev, _TOTAL_HDR, [
        ["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 100.0, 1, 1, 0.01, 100, "MB/s"],
        ["大核", "lmbench-lat", "lat_sys", "lat_syscall null", 20.0, 1, 1, 0.02, 20, "microseconds"],
    ])
    d = ld.build_digest(cur, prev_total_path=prev, top_n=8)
    assert d["ok"] and d["n_metrics"] == 2
    vp = d["vs_previous"]
    assert vp["matched"] == 2
    assert vp["improved"] == 1 and vp["regressed"] == 1  # bw↑ good, lat↑ bad
    # raw delta is +10% for both, but normalized improvement flips sign for latency
    bw = next(x for x in vp["top_improvements"] if x["command"] == "bw_mem rd")
    lat = next(x for x in vp["top_regressions"] if x["command"] == "lat_syscall null")
    assert bw["delta_pct"] == 10.0 and bw["improvement_pct"] == 10.0
    assert lat["delta_pct"] == 10.0 and lat["improvement_pct"] == -10.0


def test_hm_vs_linux_weighted_gap(tmp_path):
    f = tmp_path / "hmlx.xlsx"
    hdr = ["benchmark_module", "performance_indicator", "tool", "metric", "command",
           "HM_大核", "linux_大核", "权重_大核", "差距_大核", "得分_大核"]
    _xlsx(f, hdr, [
        ["内存", "bw test", "lmbench-mem", "bw_rd", "bw_mem rd", 110.0, 100.0, 3, "10.00%", 3],
        ["内存", "bw test2", "lmbench-mem", "bw_wr", "bw_mem wr", 90.0, 100.0, 1, "-10.00%", 1],
    ])
    d = ld.build_digest(tmp_path / "none.xlsx", hm_linux_path=f, top_n=8) if False else None
    # build_digest needs a total; make a matching minimal one
    total = tmp_path / "t.xlsx"
    _xlsx(total, _TOTAL_HDR, [["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 110, 1, 1, 0.01, 110, "MB/s"]])
    d = ld.build_digest(total, hm_linux_path=f, top_n=8)
    hv = d["hm_vs_linux"]
    # weighted gap = (3*10 + 1*-10)/(3+1) = 5.0
    assert hv["by_core"]["大核"]["weighted_gap_pct"] == 5.0
    assert hv["overall_weighted_gap_pct"] == 5.0
    # bw_wr (gap -10%, higher_better) is the regression; bw_rd (+10%) the win
    assert hv["top_regressions"][0]["command"] == "bw_mem wr"
    assert hv["top_wins"][0]["command"] == "bw_mem rd"


def test_malformed_file_does_not_raise(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_text("not an xlsx", encoding="utf-8")
    d = ld.build_digest(bad)
    assert d["ok"] is False and "error" in d


def test_lockfile_prev_degrades_to_warning(tmp_path):
    # Excel leaves a ~$<name>.xlsx companion while a workbook is open; if that
    # ever reaches us as `prev`, the digest must still succeed (just no vs_previous).
    total = tmp_path / "t.xlsx"
    _xlsx(total, _TOTAL_HDR, [["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 110, 1, 1, 0.01, 110, "MB/s"]])
    lock = tmp_path / "~$total_result_Hongmeng_20260626-154706.xlsx"
    lock.write_bytes(b"\x00\x01garbage")  # not a real xlsx, and name marks it a lock file
    d = ld.build_digest(total, prev_total_path=lock, top_n=8)
    assert d["ok"] is True
    assert "vs_previous" not in d
    assert any("lock file" in w for w in d["warnings"])


def test_locked_prev_does_not_kill_digest(tmp_path):
    # A real-but-unreadable prev (e.g. open/locked on Windows) is simulated with a
    # corrupt non-lockfile name: vs_previous is skipped via warning, digest stays ok.
    total = tmp_path / "t.xlsx"
    _xlsx(total, _TOTAL_HDR, [["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 110, 1, 1, 0.01, 110, "MB/s"]])
    corrupt = tmp_path / "total_result_Hongmeng_20260626-154706.xlsx"
    corrupt.write_text("locked/corrupt", encoding="utf-8")
    d = ld.build_digest(total, prev_total_path=corrupt, top_n=8)
    assert d["ok"] is True
    assert "vs_previous" not in d
    assert any("vs_previous skipped" in w for w in d["warnings"])


def test_percent_and_number_coercion():
    assert ld._num("1.04%") == 1.04
    assert ld._num(2.5) == 2.5
    assert ld._num("") is None and ld._num(None) is None and ld._num("n/a") is None


def test_num_rejects_non_finite():
    # 'nan'/'inf' cells must become None — NaN is invalid JSON and breaks the
    # strict MCP/JSON-RPC boundary back to the agent.
    import math
    assert ld._num("nan") is None
    assert ld._num("inf") is None and ld._num("-inf") is None
    assert ld._num(float("nan")) is None
    assert ld._num(math.inf) is None


def test_nan_gap_does_not_leak_nan(tmp_path):
    # An empty/'nan' 差距 cell used to propagate NaN into weighted_gap_pct and the
    # whole digest, making it un-serializable by a strict JSON consumer -> the
    # agent polled status forever. The digest must now be strict-JSON-safe.
    total = tmp_path / "t.xlsx"
    _xlsx(total, _TOTAL_HDR, [["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 110, 1, 1, 0.01, 110, "MB/s"]])
    hmlx = tmp_path / "hmlx.xlsx"
    hdr = ["benchmark_module", "performance_indicator", "tool", "metric", "command",
           "HM_大核", "linux_大核", "权重_大核", "差距_大核", "得分_大核"]
    _xlsx(hmlx, hdr, [
        ["内存", "bw test", "lmbench-mem", "bw_rd", "bw_mem rd", 110.0, 100.0, 3, "nan", 1.5],
    ])
    d = ld.build_digest(total, hm_linux_path=hmlx, top_n=8)
    assert d["ok"] is True
    # the nan gap is dropped: the core contributes nothing, so it never enters by_core
    assert d["hm_vs_linux"]["by_core"] == {}
    assert d["hm_vs_linux"]["overall_weighted_gap_pct"] is None
    # the decisive property: serializable by a STRICT encoder (what MCP/JSON-RPC uses)
    json.dumps(d, allow_nan=False)


def test_partial_nan_gap_keeps_valid_cores(tmp_path):
    # A nan gap on one core must not wipe out a sibling core that has a real gap.
    total = tmp_path / "t.xlsx"
    _xlsx(total, _TOTAL_HDR, [["大核", "lmbench-mem", "bw_rd", "bw_mem rd", 110, 1, 1, 0.01, 110, "MB/s"]])
    hmlx = tmp_path / "hmlx.xlsx"
    hdr = ["benchmark_module", "performance_indicator", "tool", "metric", "command",
           "HM_大核", "linux_大核", "权重_大核", "差距_大核", "得分_大核",
           "HM_中核", "linux_中核", "权重_中核", "差距_中核", "得分_中核"]
    _xlsx(hmlx, hdr, [
        ["内存", "bw test", "lmbench-mem", "bw_rd", "bw_mem rd",
         110.0, 100.0, 3, "nan", 1.5, 90.0, 100.0, 2, "10.00%", 1.0],
    ])
    d = ld.build_digest(total, hm_linux_path=hmlx, top_n=8)
    assert "大核" not in d["hm_vs_linux"]["by_core"]          # nan gap dropped
    assert d["hm_vs_linux"]["by_core"]["中核"]["weighted_gap_pct"] == 10.0
    json.dumps(d, allow_nan=False)
