#!/usr/bin/env python3
import torch, time, argparse
from src.kernels.triton.sdp_attention import sdpa_triton

def torch_sdpa(Q,K,V,scale):
    scores = torch.matmul(Q, K.transpose(-1,-2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--D", type=int, default=64)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    Q = torch.randn(args.B, args.H, args.S, args.D, device=device, dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    scale = 1.0 / (args.D ** 0.5)

    t_triton = bench(lambda: sdpa_triton(Q,K,V,scale))
    t_torch  = bench(lambda: torch_sdpa(Q,K,V,scale))

    out_t = sdpa_triton(Q,K,V,scale)
    out_p = torch_sdpa(Q,K,V,scale)
    err = (out_t - out_p).abs().max().item()

    print(f"Triton SDPA: {t_triton*1e3:.2f} ms/iter")
    print(f"PyTorch SDPA: {t_torch*1e3:.2f} ms/iter")
    print(f"Max abs diff: {err:.3e}")

if __name__ == "__main__":
    main()

