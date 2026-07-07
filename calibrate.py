"""Data-driven threshold calibration from transaction logs.

Reads the JSON files written by app/transaction_logger.py and suggests
thresholds from observed distributions instead of hand-tuned guesses:

  SCANNER_ITEM_DISTANCE   p95 of confirmed-scan distances (+10% margin)
  SCAN_MIN_DURATION       p10 of confirmed-scan evidence durations
  PAYMENT_COMPLETE_TIME   p10 of confirmed payment durations
  SCAN_COOLDOWN           p5 of gaps between consecutive scans
  CUSTOMER_DWELL_TIME     p10 of service times in clean (non-suspicious) sales

Usage:
    python calibrate.py             # print report
    python calibrate.py --write     # also update the values in .env
"""
import argparse
import glob
import json
import os
import statistics
import sys

from app import variables


def _percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def load_logs(log_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"  skipping unreadable log {path}: {e}")
    return records


def collect_samples(records):
    scan_distances, scan_durations, scan_gaps = [], [], []
    payment_durations, clean_service_times = [], []

    for rec in records:
        ev = rec.get("evidence", {})
        suspicious = rec.get("verdict", {}).get("suspicious_activity", False)

        times = []
        for scan in ev.get("scanned_items", []):
            if scan.get("distance") is not None:
                scan_distances.append(scan["distance"])
            if scan.get("duration") is not None:
                scan_durations.append(scan["duration"])
            if scan.get("time") is not None:
                times.append(scan["time"])
        times.sort()
        scan_gaps.extend(b - a for a, b in zip(times, times[1:]))

        payment_durations.extend(ev.get("payment_times", []))

        if not suspicious:
            clean_service_times.extend(ev.get("service_times", []))

    return {
        "scan_distances": scan_distances,
        "scan_durations": scan_durations,
        "scan_gaps": scan_gaps,
        "payment_durations": payment_durations,
        "clean_service_times": clean_service_times,
    }


def suggest(samples):
    """Return {ENV_NAME: (current, suggested, n_samples, note)}."""
    out = {}

    d = samples["scan_distances"]
    if d:
        out["SCANNER_ITEM_DISTANCE"] = (
            variables.SCANNER_ITEM_DISTANCE,
            round((_percentile(d, 95) or 0) * 1.1),
            len(d), "p95 of confirmed-scan distances +10%"
        )

    sd = samples["scan_durations"]
    if sd:
        out["SCAN_MIN_DURATION"] = (
            variables.SCAN_MIN_DURATION,
            round(_percentile(sd, 10), 2),
            len(sd), "p10 of confirmed-scan evidence durations"
        )

    p = samples["payment_durations"]
    if p:
        out["PAYMENT_COMPLETE_TIME"] = (
            variables.PAYMENT_COMPLETE_TIME,
            round(_percentile(p, 10), 2),
            len(p), "p10 of confirmed payment durations"
        )

    g = samples["scan_gaps"]
    if g:
        out["SCAN_COOLDOWN"] = (
            variables.SCAN_COOLDOWN,
            round(_percentile(g, 5), 2),
            len(g), "p5 of gaps between consecutive scans"
        )

    st = samples["clean_service_times"]
    if st:
        out["CUSTOMER_DWELL_TIME"] = (
            variables.CUSTOMER_DWELL_TIME,
            round(_percentile(st, 10), 2),
            len(st), "p10 of service times in clean sales"
        )

    return out


def write_env(suggestions, env_path=".env"):
    with open(env_path, encoding="utf-8") as f:
        lines = f.readlines()

    updated = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        for name, (_cur, new, _n, _note) in suggestions.items():
            if stripped.startswith(f"{name}="):
                lines[i] = f"{name}={new}\n"
                updated.add(name)

    for name, (_cur, new, _n, _note) in suggestions.items():
        if name not in updated:
            lines.append(f"{name}={new}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=variables.TRANSACTION_LOG_DIR)
    parser.add_argument("--write", action="store_true", help="update .env with suggestions")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="minimum samples per metric before trusting a suggestion")
    args = parser.parse_args()

    records = load_logs(args.log_dir)
    print(f"Loaded {len(records)} transaction logs from {args.log_dir}")
    if not records:
        sys.exit("No logs yet — run some transactions first.")

    samples = collect_samples(records)
    suggestions = suggest(samples)
    if not suggestions:
        sys.exit("Logs contain no usable evidence samples yet.")

    print(f"\n{'threshold':<24}{'current':>10}{'suggested':>10}{'n':>6}  basis")
    print("-" * 80)
    reliable = {}
    for name, (cur, new, n, note) in suggestions.items():
        flag = "" if n >= args.min_samples else f"  (LOW SAMPLE COUNT, need {args.min_samples})"
        print(f"{name:<24}{cur:>10}{new:>10}{n:>6}  {note}{flag}")
        if n >= args.min_samples:
            reliable[name] = suggestions[name]

    if args.write:
        if not reliable:
            sys.exit("\nNothing written: no metric has enough samples yet.")
        write_env(reliable)
        print(f"\nWrote {len(reliable)} threshold(s) to .env: {', '.join(reliable)}")
    else:
        print("\nDry run — pass --write to apply reliable suggestions to .env")


if __name__ == "__main__":
    main()
