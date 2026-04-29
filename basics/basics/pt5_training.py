import argparse
import array
import math
import torch
import torch.nn as nn
import numpy as np
from pt3_linear import TransformerLM
import json
import time
from pathlib import Path


def data_loading(x: array, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    returned pair of tensors: sampled input seq and corresponding next token targets
    tensors have shape (batch_size, context_length), containing token IDs, placed on device
    '''
    starts = np.random.randint(0, len(x) - context_length, size=(batch_size,))
    input_batch = np.stack([x[start:start+context_length] for start in starts])
    target_batch = np.stack([x[start+1:start+context_length+1] for start in starts])
    input = torch.tensor(input_batch, device=device)
    target = torch.tensor(target_batch, device=device)
    return input, target

def get_batch(data, batch_size, context_length, device):
    """
    Sample random subsequences from a 1D token array.

    data: np.memmap or numpy array of shape (N,)
    returns:
        x: (B, T)
        y: (B, T) where y is next-token targets
    """
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x_batch = np.stack([data[i:i + context_length] for i in starts])
    y_batch = np.stack([data[i + 1:i + 1 + context_length] for i in starts])

    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, context_length, device):
    """
    Compute average train/val loss over a few batches.
    """
    model.eval()
    out = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)  # (B, T, vocab_size)

            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, V),
                y.view(B * T)
            )
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def save_checkpoint(model, optimizer, step, best_val_loss, save_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)

    # model hyperparameters
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=128)

    # optimization hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_iters", type=int, default=5000)

    # logging / eval / checkpointing
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="checkpoint.pt")

    # optional
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)

    # memory-efficient loading with memmap
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    # build model
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        device=device,
        dtype=torch.float32,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    model.train()
    for step in range(args.max_iters):
        # sample batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # forward
        logits = model(x)  # (B, T, V)
        B, T, V = logits.shape

        loss = nn.functional.cross_entropy(
            logits.view(B * T, V),
            y.view(B * T)
        )

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # logging
        if step % args.log_interval == 0:
            print(f"step {step:5d} | train loss {loss.item():.4f}")

        # eval + checkpoint
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )

            print(
                f"[eval] step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    save_path=args.save_path,
                )
                print(f"saved best checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()