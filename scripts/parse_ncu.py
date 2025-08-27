#!/usr/bin/env python3
import argparse, csv, json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, newline='')))
    # print top by sm__throughput if present
    key = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
    rows = [r for r in rows if key in r and r[key]]
    rows.sort(key=lambda r: float(r[key]), reverse=True)
    print(json.dumps(rows[:10], indent=2))
if __name__ == "__main__":
    main()


