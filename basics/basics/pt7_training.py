import argparse
import array
import math
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from pt3_linear import TransformerLM
from pt4_crossentropy import cross_entropy
from pt2_tokenizer import Tokenizer
from pt5_training import get_batch


import json, glob
import pandas as pd

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
            loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
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


def get_grad_norm(model):
    return sum(
        p.grad.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5


def get_weight_norm(model):
    return sum(
        p.norm(2).item() ** 2
        for p in model.parameters()
    ) ** 0.5


def register_activation_hooks(model):
    """Register forward hooks on each transformer layer to track activation norms."""
    activation_norms = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_norms[name] = output.norm(2).item()
        return hook

    # Try to hook into named children — adjust if your model structure differs
    for name, module in model.named_modules():
        if name:  # skip root module
            module.register_forward_hook(make_hook(name))

    return activation_norms


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)

    # model hyperparameters
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)

    # optimization hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_iters", type=int, default=5000)

    # logging / eval / checkpointing
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="checkpoint.pt")
    parser.add_argument("--run_name", type=str, default="run")

    # debugging
    parser.add_argument("--overfit_single_batch", action="store_true",
                        help="Fix a single batch and overfit to it (sanity check)")
    parser.add_argument("--monitor_norms", action="store_true",
                        help="Log gradient, weight, and activation norms")

    # optional
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    class Args:
        train_data = "tinystories_train_ids.bin"
        val_data   = "tinystories_valid_ids.bin"
        vocab_size = 10000
        d_model = 512
        num_heads = 8
        d_ff = 2048
        num_layers = 4
        context_length = 256

        batch_size = 64
        learning_rate = 1e-3
        weight_decay = 0.1
        max_iters = 2500

        eval_interval = 200
        eval_iters = 100
        log_interval = 50
        save_path = "checkpoint.pt"
        run_name = "colab_run"

        overfit_single_batch = False
        monitor_norms = False

        device = "cpu"

    args = Args()
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
    reached_target = False

    # --- experiment logging setup ---
    config_dict = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "log_interval": args.log_interval,
        "save_path": args.save_path,
        "run_name": args.run_name,
        "overfit_single_batch": args.overfit_single_batch,
        "monitor_norms": args.monitor_norms,
        "device": args.device,
    }

    log = {
        "run_name": args.run_name,
        "config": config_dict,
        "steps_train": [],
        "steps_eval": [],
    }

    start_time = time.time()
    save_path = args.save_path
    log_path = Path(args.run_name + "_log.json")

    # --- activation norm hooks ---
    activation_norms = {}
    if args.monitor_norms:
        activation_norms = register_activation_hooks(model)

    # --- single batch overfitting ---
    if args.overfit_single_batch:
        print("Overfitting to a single batch...")
        x_fixed, y_fixed = get_batch(train_data, args.batch_size, args.context_length, device)

    model.train()
    for step in range(args.max_iters):

        # sample batch
        if args.overfit_single_batch:
            x, y = x_fixed, y_fixed
        else:
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # forward
        logits = model(x)  # (B, T, V)
        B, T, V = logits.shape
        loss = cross_entropy(logits.view(B * T, V), y.view(B * T))

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # compute norms after backward
        grad_norm = get_grad_norm(model) if args.monitor_norms else None
        weight_norm = get_weight_norm(model) if args.monitor_norms else None

        optimizer.step()

        # logging
        if step % args.log_interval == 0:
            wallclock = time.time() - start_time
            log_entry = {
                "step": step,
                "train_loss": loss.item(),
                "wallclock_time": wallclock,
            }
            if args.monitor_norms:
                log_entry["grad_norm"] = grad_norm
                log_entry["weight_norm"] = weight_norm
                # log a few key activation norms (avoid logging all to keep file small)
                log_entry["activation_norms"] = {
                    k: v for k, v in list(activation_norms.items())[:5]
                }

            log["steps_train"].append(log_entry)

            norm_str = ""
            if args.monitor_norms:
                norm_str = f" | grad_norm {grad_norm:.4f} | weight_norm {weight_norm:.4f}"
            print(f"step {step:5d} | train loss {loss.item():.4f}{norm_str}")

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

            wallclock = time.time() - start_time

            log["steps_eval"].append({
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "wallclock_time": wallclock,
            })

            # write log to disk after every eval
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

            print(
                f"[eval] step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"time {wallclock:.1f}s"
            )

            # check target val loss
            if losses["val"] <= 2.0 and not reached_target:
                reached_target = True
                print(f"Target val loss of 2.0 reached at step {step}!")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    save_path=save_path,
                )
                print(f"Saved best checkpoint to {save_path} (val loss {best_val_loss:.4f})")

    print(f"Training complete. Log saved to {log_path}")
    print(f"Best val loss: {best_val_loss:.4f}")

    rows = []
    for path in sorted(glob.glob("*_log.json")):
        with open(path) as f:
            log = json.load(f)
        
        cfg = log["config"]
        evals = log["steps_eval"]
        
        best = min(e["val_loss"] for e in evals)
        final = evals[-1]
        
        rows.append({
            "run_name":      log["run_name"],
            "best_val_loss": round(best, 4),
            "final_val":     round(final["val_loss"], 4),
            "final_train":   round(final["train_loss"], 4),
            # hyperparams
            "lr":            cfg["learning_rate"],
            "d_model":       cfg["d_model"],
            "num_layers":    cfg["num_layers"],
            "d_ff":          cfg["d_ff"],
            "weight_decay":  cfg["weight_decay"],
            "batch_size":    cfg["batch_size"],
            "max_iters":     cfg["max_iters"],
        })

    df = pd.DataFrame(rows).sort_values("best_val_loss")
    df

if __name__ == "__main__":
    main()