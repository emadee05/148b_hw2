import argparse
import torch

from basics.model import BasicsTransformerLM


MODEL_SIZES = {
    "small":  {"d_model": 512,  "d_ff": 2048, "num_layers": 8,  "num_heads": 8},
    "medium": {"d_model": 768,  "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "large":  {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_memory_profile(
    size: str,
    context_length: int,
    mode: str,
    mixed_precision: bool,
    output_file: str,
):
    device = "cuda"
    vocab_size = 10000
    batch_size = 4

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Warmup: do NOT record memory history here.
    for _ in range(5):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=mixed_precision,
        ):
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        sync()

    torch.cuda.empty_cache()
    sync()

    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=mixed_precision,
    ):
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    if mode == "train":
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    sync()

    # Save snapshot.
    torch.cuda.memory._dump_snapshot(output_file)

    # Stop recording memory history.
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    print(f"Saved memory snapshot to: {output_file}")
    print(f"Model size: {size}")
    print(f"Context length: {context_length}")
    print(f"Mode: {mode}")
    print(f"Mixed precision BF16: {mixed_precision}")
    print(f"Peak allocated memory: {peak_allocated:.2f} MB")
    print(f"Peak reserved memory: {peak_reserved:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], default="large")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--mode", choices=["forward", "train"], default="forward")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    if args.output_file is None:
        mp = "bf16" if args.mixed_precision else "fp32"
        args.output_file = (
            f"memory_{args.size}_ctx{args.context_length}_{args.mode}_{mp}.pickle"
        )

    run_memory_profile(
        size=args.size,
        context_length=args.context_length,
        mode=args.mode,
        mixed_precision=args.mixed_precision,
        output_file=args.output_file,
    )
