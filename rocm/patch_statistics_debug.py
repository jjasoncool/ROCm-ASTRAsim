"""
patch_statistics_debug.py
Patch Statistics.cc to add COMM interval debug logging.
Read-only instrumentation — does NOT change simulation logic.

Usage:
    python3 patch_statistics_debug.py /path/to/astra-sim/astra-sim/workload/Statistics.cc
"""
import sys

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

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/astra-sim/astra-sim/workload/Statistics.cc"
    with open(path, "r") as f:
        src = f.read()
    if FIND not in src:
        print(f"[ERROR] Pattern not found in {path} — file may already be patched or upstream changed", file=sys.stderr)
        sys.exit(1)
    count = src.count(FIND)
    if count > 1:
        print(f"[ERROR] Pattern found {count} times — expected exactly 1", file=sys.stderr)
        sys.exit(1)
    with open(path, "w") as f:
        f.write(src.replace(FIND, REPLACE))
    print(f"[OK] Patched {path} — added COMM interval debug logging")