import argparse
import timeit
import torch
import numpy as np

from basics.model import BasicsTransformerLM


MODEL_SIZES = {
    "small":  {"d_model": 512,  "d_ff": 2048, "num_layers": 8,  "num_heads": 8},
    "medium": {"d_model": 768,  "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "large":  {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(size, warmup_steps=5, measure_steps=10, mode="both", mixed_precision=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 10000
    batch_size = 4
    context_length = 128

    cfg = MODEL_SIZES[size]

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device)

    model.train()

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    loss_fn = torch.nn.CrossEntropyLoss()

    autocast_enabled = mixed_precision and device == "cuda"

    def forward_and_loss():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        return logits, loss

    def backward_pass(loss):
        loss.backward()
        model.zero_grad(set_to_none=True)

    # warmup (not measured)
    for _ in range(warmup_steps):
        logits, loss = forward_and_loss()
        if mode in ["backward", "both"]:
            backward_pass(loss)
        sync()

    forward_times = []
    backward_times = []

    for _ in range(measure_steps):
        sync()
        t0 = timeit.default_timer()

        logits, loss = forward_and_loss()

        sync()
        t1 = timeit.default_timer()

        if mode in ["forward", "both"]:
            forward_times.append(t1 - t0)

        if mode in ["backward", "both"]:
            sync()
            t2 = timeit.default_timer()

            backward_pass(loss)

            sync()
            t3 = timeit.default_timer()

            backward_times.append(t3 - t2)

    print(f"\nModel size: {size}")
    print(f"Device: {device}")
    print(f"Mixed precision BF16: {mixed_precision}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Measurement steps: {measure_steps}")

    if forward_times:
        print(f"Forward avg: {np.mean(forward_times):.6f} s")
        print(f"Forward std: {np.std(forward_times):.6f} s")

    if backward_times:
        print(f"Backward avg: {np.mean(backward_times):.6f} s")
        print(f"Backward std: {np.std(backward_times):.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], required=True)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "backward", "both"], default="both")
    parser.add_argument("--mixed_precision", action="store_true")

    args = parser.parse_args()

    benchmark(
        size=args.size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        mixed_precision=args.mixed_precision,
    )
