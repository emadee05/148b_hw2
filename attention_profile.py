import itertools
import time
import csv
import traceback

import torch
import torch.nn.functional as F


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def naive_attention(Q, K, V):
    """
    Q, K, V shape: (batch_size, seq_len, d_model)
    single-head causal scaled dot-product attention
    """
    batch_size, seq_len, d_model = Q.shape

    scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)

    # causal mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = attn @ V
    return out


def benchmark_one(batch_size, seq_len, d_model, num_iters=100, warmup_iters=10):
    device = "cuda"
    dtype = torch.float32

    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(warmup_iters):
        out = naive_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        sync()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync()

    # Time forward passes
    forward_start = time.perf_counter()

    out = None
    loss = None

    for _ in range(num_iters):
        out = naive_attention(Q, K, V)
        loss = out.sum()
        sync()

    forward_end = time.perf_counter()

    # Memory right before backward starts
    memory_before_backward_mb = torch.cuda.memory_allocated() / 1024**2
    peak_forward_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Time backward passes
    backward_start = time.perf_counter()

    for _ in range(num_iters):
        loss.backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        sync()

    backward_end = time.perf_counter()

    peak_total_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    forward_time_ms = (forward_end - forward_start) / num_iters * 1000
    backward_time_ms = (backward_end - backward_start) / num_iters * 1000

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "forward_time_ms": forward_time_ms,
        "backward_time_ms": backward_time_ms,
        "memory_before_backward_mb": memory_before_backward_mb,
        "peak_forward_memory_mb": peak_forward_memory_mb,
        "peak_total_memory_mb": peak_total_memory_mb,
        "status": "ok",
        "error": "",
    }


def main():
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [64, 128, 256, 512, 1024]

    results = []

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        print(f"\nRunning d_model={d_model}, seq_len={seq_len}")

        try:
            result = benchmark_one(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_iters=100,
                warmup_iters=10,
            )
            print(
                f"forward={result['forward_time_ms']:.3f} ms, "
                f"backward={result['backward_time_ms']:.3f} ms, "
                f"mem_before_backward={result['memory_before_backward_mb']:.2f} MB, "
                f"peak={result['peak_total_memory_mb']:.2f} MB"
            )

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            result = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "d_model": d_model,
                "forward_time_ms": None,
                "backward_time_ms": None,
                "memory_before_backward_mb": None,
                "peak_forward_memory_mb": None,
                "peak_total_memory_mb": None,
                "status": "OOM",
                "error": str(e).replace("\n", " "),
            }
            print("OOM")

        except Exception as e:
            torch.cuda.empty_cache()
            result = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "d_model": d_model,
                "forward_time_ms": None,
                "backward_time_ms": None,
                "memory_before_backward_mb": None,
                "peak_forward_memory_mb": None,
                "peak_total_memory_mb": None,
                "status": "error",
                "error": traceback.format_exc().replace("\n", " "),
            }
            print("ERROR:", e)

        results.append(result)

    output_csv = "attention_profile_results.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    main()
