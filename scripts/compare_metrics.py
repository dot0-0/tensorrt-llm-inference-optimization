#!/usr/bin/env python3
import json, argparse

def load(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pytorch', default='metrics_pytorch.json')
    ap.add_argument('--trt', default='metrics_trtllm.json')
    args = ap.parse_args()

    pt = load(args.pytorch)
    tr = load(args.trt)

    tps_pt, tps_tr = pt['tokens_per_sec'], tr['tokens_per_sec']
    tps_gain = tps_tr / tps_pt if tps_pt > 0 else 0.0

    # Estimate per-token latency from per-batch stats
    def per_token_lat(per_batch):
        vals = []
        for b in per_batch:
            if b['new_tokens'] > 0 and b['seconds'] > 0:
                vals.append(b['seconds'] / b['new_tokens'])
        return sum(vals)/len(vals) if vals else 0.0

    lat_pt = per_token_lat(pt['per_batch'])
    lat_tr = per_token_lat(tr['per_batch'])
    lat_reduction = (1.0 - (lat_tr / lat_pt)) * 100.0 if lat_pt > 0 else 0.0

    print(f"Throughput gain (tokens/sec): {tps_tr:.3f} / {tps_pt:.3f} = {tps_gain:.2f}×")
    print(f"Per-token latency reduction: {lat_reduction:.1f}%  (PT={lat_pt*1e3:.3f} ms, TRT={lat_tr*1e3:.3f} ms)")

if __name__ == '__main__':
    main()

