#!/usr/bin/env python3
import argparse, os, subprocess, json, tempfile, pathlib, time, csv

def make_prompts(token_len: int, count: int):
    # crude token proxy: repeat short words
    s = ("hello " * max(1, token_len)).strip()
    return [s for _ in range(count)]

def write_prompts_file(lines, path):
    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")

def run_pytorch(model, dtype, batch_size, max_new, prompts_file, out_json):
    cmd = [
        "python", "scripts/benchmark_pytorch.py",
        "--model", model, "--dtype", dtype,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new),
        "--prompts_file", prompts_file,
        "--out", out_json
    ]
    subprocess.check_call(cmd)
    return json.load(open(out_json))

def run_trt(engine_dir, max_new, prompts_file, out_json):
    cmd = [
        "trtllm-run",
        "--engine_dir", engine_dir,
        "--prompts_file", prompts_file,
        "--max_output_len", str(max_new),
        "--report_json", out_json
    ]
    subprocess.check_call(cmd)
    return json.load(open(out_json))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["pytorch","trtllm","both"], default="both")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--engine_dir", default="build/llama2-7b-bf16")
    ap.add_argument("--batch_sizes", default="1,2,4")
    ap.add_argument("--prompt_lens", default="64,512,1024")
    ap.add_argument("--max_new_tokens", default="32,64")
    ap.add_argument("--batches_per_point", type=int, default=4, help="how many batches to run per point")
    ap.add_argument("--out_csv", default="data/matrix_results.csv")
    args = ap.parse_args()

    bs_list = [int(x) for x in args.batch_sizes.split(",") if x]
    pl_list = [int(x) for x in args.prompt_lens.split(",") if x]
    mn_list = [int(x) for x in args.max_new_tokens.split(",") if x]

    pathlib.Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["framework","batch_size","prompt_len","max_new_tokens","tokens_per_sec","per_token_latency_avg_s"])

        for bs in bs_list:
            for pl in pl_list:
                for mn in mn_list:
                    # build prompts file: batches_per_point * batch_size lines
                    num_prompts = args.batches_per_point * bs
                    prompts = make_prompts(pl, num_prompts)
                    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                        prompts_file = tmp.name
                    write_prompts_file(prompts, prompts_file)

                    if args.framework in ("pytorch","both"):
                        out_json = f"tmp_pt_{bs}_{pl}_{mn}.json"
                        pt = run_pytorch(args.model, args.dtype, bs, mn, prompts_file, out_json)
                        # derive avg per-token latency
                        lats = [b["seconds"]/b["new_tokens"] for b in pt["per_batch"] if b["new_tokens"]>0 and b["seconds"]>0]
                        avg_lat = sum(lats)/len(lats) if lats else 0.0
                        w.writerow(["pytorch", bs, pl, mn, pt["tokens_per_sec"], avg_lat])

                    if args.framework in ("trtllm","both"):
                        out_json = f"tmp_trt_{bs}_{pl}_{mn}.json"
                        tr = run_trt(args.engine_dir, mn, prompts_file, out_json)
                        lats = [b["seconds"]/b["new_tokens"] for b in tr["per_batch"] if b["new_tokens"]>0 and b["seconds"]>0]
                        avg_lat = sum(lats)/len(lats) if lats else 0.0
                        w.writerow(["tensorrt-llm", bs, pl, mn, tr["tokens_per_sec"], avg_lat])

                    os.unlink(prompts_file)

    print("[OK] Wrote", args.out_csv)

if __name__ == "__main__":
    main()

