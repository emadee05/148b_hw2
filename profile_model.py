import argparse
import torch
import torch.cuda.nvtx as nvtx

from basics.model import BasicsTransformerLM
from basics.pt3_linear import annotated_scaled_dot_product_attention
import basics.basics.model as model_module

model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention


MODEL_SIZES = {
    "small":  {"d_model": 512,  "d_ff": 2048, "num_layers": 8,  "num_heads": 8},
    "medium": {"d_model": 768,  "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "large":  {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(size, context_length, mode):
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

    # warmup: do NOT profile this
    for _ in range(5):
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()

    # profiled section
    sync()
    nvtx.range_push("profiled_step")

    if mode in ["forward", "train"]:
        nvtx.range_push("forward")
        logits = model(x)
        nvtx.range_pop()

    if mode in ["backward", "train"]:
        nvtx.range_push("loss")
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

    if mode == "train":
        nvtx.range_push("optimizer_step")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        nvtx.range_pop()

    sync()
    nvtx.range_pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--mode", choices=["forward", "backward", "train"], default="train")
    args = parser.parse_args()

    main(args.size, args.context_length, args.mode)
