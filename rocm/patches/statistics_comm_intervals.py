#!/usr/bin/env python3
"""Log one line per COMM interval in ASTRA-sim's statistics pass.

Read-only: logs intervals that are already being reduced, does not change
type_time or any simulation result. ASTRA-sim otherwise reports only the merged
COMM total, so per-collective cost cannot be recovered from a run.

Usage: python3 statistics_comm_intervals.py <astra-sim-root>
"""
import sys

REL = "astra-sim/workload/Statistics.cc"
MARKER = "[DEBUG] COMM interval"

FIND = """\
    this->type_time.clear();
    for (const auto& [type, intervals] : interval_map) {
        this->type_time[type] = _calculateTotalRuntimeFromIntervals(intervals);
    }"""

REPLACE = """\
    auto logger = LoggerFactory::get_logger("statistics");
    this->type_time.clear();
    for (const auto& [type, intervals] : interval_map) {
        if (type == OperatorStatistics::OperatorType::COMM) {
            auto sorted = intervals;
            std::sort(sorted.begin(), sorted.end());
            Tick raw_sum = 0;
            for (size_t i = 0; i < sorted.size(); i++) {
                Tick dur = sorted[i].second - sorted[i].first;
                raw_sum += dur;
                logger->info("[DEBUG] COMM interval[{}]: start={} end={} dur={}",
                    i, sorted[i].first, sorted[i].second, dur);
            }
            logger->info("[DEBUG] COMM total_intervals={} raw_sum={}", sorted.size(), raw_sum);
        }
        this->type_time[type] = _calculateTotalRuntimeFromIntervals(intervals);
        if (type == OperatorStatistics::OperatorType::COMM) {
            logger->info("[DEBUG] COMM merged_result={}", this->type_time[type]);
        }
    }"""

path = f"{sys.argv[1] if len(sys.argv) > 1 else '/workspace/astra-sim'}/{REL}"
try:
    with open(path, newline="") as f:
        src = f.read()
except OSError as e:
    sys.exit(f"[statistics_comm_intervals] ERROR: {e}")

if MARKER in src:
    print(f"[statistics_comm_intervals] already patched: {REL}")
    sys.exit(0)

n = src.count(FIND)
if n != 1:
    sys.exit(f"[statistics_comm_intervals] ERROR: anchor found {n} times, expected 1 "
             f"(upstream changed?) in {REL}")

with open(path, "w", newline="") as f:
    f.write(src.replace(FIND, REPLACE))
print(f"[statistics_comm_intervals] patched: {REL}")
