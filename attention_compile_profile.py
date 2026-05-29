import itertools
import time
import csv
import traceback
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def naive_attention(Q, K, V):
    _, seq_len, d_model = Q.shape

    scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)

    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
        diagonal=1,
    )

    scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ V
    return out


def clear_grads(Q, K, V):
    Q.grad = None
    K.grad = None
    V.grad = None


def benchmark_attention(attn_fn, batch_size, seq_len, d_model, num_iters=100, warmup_iters=10):
    device = "cuda"
    dtype = torch.float32

    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(warmup_iters):
        out = attn_fn(Q, K, V)
        loss = out.sum()
        loss.backward()
        clear_grads(Q, K, V)
        sync()

    # Forward timing
    forward_times = []
    for _ in range(num_iters):
        sync()
        t0 = time.perf_counter()

        out = attn_fn(Q, K, V)

        sync()
        t1 = time.perf_counter()
        forward_times.append((t1 - t0) * 1000)

    # Backward timing: fresh forward each time
    backward_times = []
    for _ in range(num_iters):
        clear_grads(Q, K, V)

        out = attn_fn(Q, K, V)
        loss = out.sum()

        sync()
        t0 = time.perf_counter()

        loss.backward()

        sync()
        t1 = time.perf_counter()
        backward_times.append((t1 - t0) * 1000)

    return {
        "forward_avg_ms": float(sum(forward_times) / len(forward_times)),
        "backward_avg_ms": float(sum(backward_times) / len(backward_times)),
    }


def main():
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [64, 128, 256, 512, 1024]

    vanilla_attention = naive_attention
    compiled_attention = torch.compile(naive_attention)

    results = []

    for version_name, attn_fn in [
        ("vanilla", vanilla_attention),
        ("compiled", compiled_attention),
    ]:
        print(f"\n===== {version_name.upper()} ATTENTION =====")

        for d_model, seq_len in itertools.product(d_models, seq_lens):
            print(f"\nRunning {version_name}: d_model={d_model}, seq_len={seq_len}")

            try:
                result = benchmark_attention(
                    attn_fn=attn_fn,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    d_model=d_model,
                    num_iters=100,
                    warmup_iters=10,
                )

                row = {
                    "version": version_name,
                    "batch_size": batch_size,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    **result,
                    "status": "ok",
                    "error": "",
                }

                print(
                    f"forward={row['forward_avg_ms']:.3f} ms, "
                    f"backward={row['backward_avg_ms']:.3f} ms"
                )

            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                row = {
                    "version": version_name,
                    "batch_size": batch_size,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_avg_ms": None,
                    "backward_avg_ms": None,
                    "status": "OOM",
                    "error": str(e).replace("\n", " "),
                }
                print("OOM")

            except Exception as e:
                torch.cuda.empty_cache()
                row = {
                    "version": version_name,
                    "batch_size": batch_size,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_avg_ms": None,
                    "backward_avg_ms": None,
                    "status": "error",
                    "error": traceback.format_exc().replace("\n", " "),
                }
                print("ERROR:", e)

            results.append(row)

    with open("attention_compile_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to attention_compile_results.csv")


if __name__ == "__main__":
    main()
