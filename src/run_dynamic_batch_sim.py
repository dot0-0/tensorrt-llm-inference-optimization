#!/usr/bin/env python3
import argparse, csv
from src.dynamic_batch_sim import simulate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrival_qps", default="25,50,100")
    ap.add_argument("--max_delay_ms", default="5,20,50")
    ap.add_argument("--batch_limit", default="1,2,4,8")
    ap.add_argument("--service_tps", type=int, default=1500)
    ap.add_argument("--req_tokens", type=int, default=64)
    ap.add_argument("--duration_s", type=int, default=15)
    ap.add_argument("--out_csv", default="data/dyn_batch_sweep.csv")
    args = ap.parse_args()

    qps = [int(x) for x in args.arrival_qps.split(",")]
    dlys = [int(x) for x in args.max_delay_ms.split(",")]
    blims = [int(x) for x in args.batch_limit.split(",")]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qps","max_delay_ms","batch_limit","p50_ms","p90_ms","avg_ms"])
        for q in qps:
            for d in dlys:
                for b in blims:
                    r = simulate(arrival_rate_qps=q, max_delay_ms=d, batch_limit=b, service_tps=args.service_tps, req_tokens=args.req_tokens, duration_s=args.duration_s)
                    w.writerow([q,d,b, r["p50_s"]*1000, r["p90_s"]*1000, r["avg_s"]*1000])
    print("[OK] Wrote", args.out_csv)

if __name__ == "__main__":
    main()

