#!/system/bin/sh
set -eu

TEST_CASE="${1:-basic_swipe}"
OUT_FILE="${2:-/data/local/tmp/basic_swipe.result}"
DURATION_S="${3:-60}"
SWIPE_COUNT="${4:-1050}"

RECORD_FILE="/data/local/tmp/${TEST_CASE}.hiperf.data"
REPORT_FILE="/data/local/tmp/${TEST_CASE}.hiperf.report"
STATUS_NOTE="Basic swipe case from HMOPT auto-test MCP"

echo "$STATUS_NOTE" > "$OUT_FILE"
echo "startswipe" >> "$OUT_FILE"
echo "test_case=$TEST_CASE" >> "$OUT_FILE"
echo "duration_s=$DURATION_S" >> "$OUT_FILE"
echo "swipe_count=$SWIPE_COUNT" >> "$OUT_FILE"

echo stopsink > /sys/class/hw_power/charger/charge_data/plugusb || true
echo 100000 > /proc/sys/kernel/perf_event_max_sample_rate || true
cat /proc/sys/kernel/perf_event_max_sample_rate >> "$OUT_FILE" || true

(
  for i in $(seq 1 "$SWIPE_COUNT"); do
    uinput -T -m 600 2000 50 2000 100 -k 1000
    uinput -T -m 600 2000 1150 2000 100 -k 1000
  done
) &
SWIPE_PID=$!

if command -v hiperf >/dev/null 2>&1; then
  if hiperf record -f 1000 -d "$DURATION_S" -a --cpu-limit 100 -e hw-instructions --kernel-callchain --enable-debuginfo-symbolic --exclude-hiperf -o "$RECORD_FILE" >> "$OUT_FILE" 2>&1; then
    if hiperf report -i "$RECORD_FILE" -o "$REPORT_FILE" >> "$OUT_FILE" 2>&1; then
      cat "$REPORT_FILE" >> "$OUT_FILE" || true
    else
      echo "warning: hiperf report failed" >> "$OUT_FILE"
    fi
  else
    echo "warning: hiperf record failed" >> "$OUT_FILE"
  fi
else
  echo "warning: hiperf not found; skipped profiling" >> "$OUT_FILE"
fi

wait "$SWIPE_PID" || true

echo startsink > /sys/class/hw_power/charger/charge_data/plugusb || true
rm -f "$RECORD_FILE" "$REPORT_FILE"

echo "end" >> "$OUT_FILE"
