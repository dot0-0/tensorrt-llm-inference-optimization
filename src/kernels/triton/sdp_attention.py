import torch, triton, triton.language as tl

@triton.jit
def _rowmax(x_ptr, stride, n_cols, out_ptr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = row_id * stride + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=tl.arange(0, BLOCK) < n_cols, other=-1e9)
    m = tl.max(x, axis=0)
    tl.store(out_ptr + row_id, m)

@triton.jit
def _rowsumexp(x_ptr, m_ptr, stride, n_cols, out_ptr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = row_id * stride + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=tl.arange(0, BLOCK) < n_cols, other=-1e9)
    m = tl.load(m_ptr + row_id)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row_id, s)

@triton.jit
def _softmax(x_ptr, m_ptr, s_ptr, stride, n_cols, out_ptr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = row_id * stride + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=tl.arange(0, BLOCK) < n_cols, other=-1e9)
    m = tl.load(m_ptr + row_id)
    s = tl.load(s_ptr + row_id)
    y = tl.exp(x - m) / s
    tl.store(out_ptr + offs, y, mask=tl.arange(0, BLOCK) < n_cols)

def triton_softmax(x):
    # x: [M, N]
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    m = torch.empty((M,), device=x.device, dtype=x.dtype)
    s = torch.empty((M,), device=x.device, dtype=x.dtype)
    grid = lambda META: (M,)
    _rowmax[grid](x, x.stride(0), N, m, BLOCK=BLOCK)
    _rowsumexp[grid](x, m, x.stride(0), N, s, BLOCK=BLOCK)
    _softmax[grid](x, m, s, x.stride(0), N, y, BLOCK=BLOCK)
    return y

def sdpa_triton(Q, K, V, scale):
    # Q,K,V: [B, H, S, D] single-head if H==1. For demo, handle B*H as batch rows.
    B, H, S, D = Q.shape
    Q2 = Q.reshape(B*H, S, D)
    K2 = K.reshape(B*H, S, D)
    V2 = V.reshape(B*H, S, D)
    scores = torch.matmul(Q2, K2.transpose(1,2)) * scale   # [B*H, S, S]
    probs = triton_softmax(scores)                         # [B*H, S, S]
    out = torch.matmul(probs, V2)                          # [B*H, S, D]
    return out.reshape(B, H, S, D)

