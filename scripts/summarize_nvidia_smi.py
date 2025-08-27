#!/usr/bin/env python3
import argparse, csv, pathlib, json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    peaks = {}
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            idx = row["index"]
            used = row["memory.used [MiB]"] if "memory.used [MiB]" in row else row.get("memory.used", "0")
            try:
                used = int(used.split()[0])
            except:
                used = 0
            peaks[idx] = max(peaks.get(idx, 0), used)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        w = csv.writer(f); w.writerow(["gpu_index","peak_mib"])
        for k,v in sorted(peaks.items(), key=lambda x:int(x[0])):
            w.writerow([k, v])
    print(json.dumps({"peaks_mib": peaks}, indent=2))
if __name__ == "__main__":
    main()

