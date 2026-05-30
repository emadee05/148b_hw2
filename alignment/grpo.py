from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
import random
from tqdm import tqdm

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    input_ids_list = []
    labels_list = []
    response_mask_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        full_sequence = prompt_ids + output_ids

        # Causal LM shift:
        # input_ids predict labels
        input_ids = full_sequence[:-1]
        labels = full_sequence[1:]

        prompt_len = len(prompt_ids)
        response_len = len(output_ids)

        # Mask aligns with labels.
        # The first prompt_len - 1 labels are still prompt tokens.
        # Then all response_len labels are response tokens.
        response_mask = [False] * (prompt_len - 1) + [True] * response_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        response_mask_list.append(response_mask)

    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = []
    padded_labels = []
    padded_response_mask = []

    for input_ids, labels, response_mask in zip(
        input_ids_list,
        labels_list,
        response_mask_list,
    ):
        pad_len = max_len - len(input_ids)

        padded_input_ids.append(input_ids + [pad_id] * pad_len)
        padded_labels.append(labels + [pad_id] * pad_len)
        padded_response_mask.append(response_mask + [False] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_response_mask, dtype=torch.bool),
    }

def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    outputs = model(input_ids)
    logits = outputs.logits  # (B, T, vocab_size)

    # Convert logits to log probabilities over vocab.
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, vocab_size)

    # Pick out the log-probability of the actual label token at each position.
    # labels shape: (B, T)
    # labels.unsqueeze(-1): (B, T, 1)
    # gathered: (B, T, 1), then squeeze -> (B, T)
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result
def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    masked_tensor = tensor * mask.to(tensor.dtype)

    if dim is None:
        return masked_tensor.sum() / normalize_constant
    else:
        return masked_tensor.sum(dim=dim) / normalize_constant

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""

    reward_infos: list[dict[str, float]] = []
    raw_rewards_list: list[float] = []
    format_rewards_list: list[float] = []
    answer_rewards_list: list[float] = []

    # 1. Compute raw rewards for each rollout response.
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, ground_truth)
        reward_infos.append(reward_info)

        # Be slightly robust to either underscore or space-style keys.
        raw_reward = reward_info.get("reward", reward_info.get("total_reward", 0.0))
        format_reward = reward_info.get("format_reward", reward_info.get("format reward", 0.0))
        answer_reward = reward_info.get("answer_reward", reward_info.get("answer reward", 0.0))

        raw_rewards_list.append(float(raw_reward))
        format_rewards_list.append(float(format_reward))
        answer_rewards_list.append(float(answer_reward))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # 2. Reshape into groups: (num_groups, group_size).
    grouped_rewards = raw_rewards.view(-1, group_size)

    # 3. Compute per-group mean.
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    # 4. Advantage = reward - group mean.
    grouped_advantages = grouped_rewards - group_means

    # 5. Optionally divide by per-group std.
    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True, unbiased=False)
        grouped_advantages = grouped_advantages / (group_stds + advantage_eps)

    advantages = grouped_advantages.reshape(-1)

    metadata: dict[str, float] = {
        "mean_reward": float(raw_rewards.mean().item()),
        "std_reward": float(raw_rewards.std(unbiased=False).item()),
        "min_reward": float(raw_rewards.min().item()),
        "max_reward": float(raw_rewards.max().item()),
        "mean_format_reward": float(torch.tensor(format_rewards_list).mean().item()),
        "mean_answer_reward": float(torch.tensor(answer_rewards_list).mean().item()),
    }

    return advantages, raw_rewards, metadata

def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)  # (B, 1)

    # Importance ratio:
    # pi_theta(token) / pi_old(token)
    ratio = torch.exp(policy_log_probs - old_log_probs)

    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - cliprange,
        max=1.0 + cliprange,
    )

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages

    # GRPO/PPO maximizes min(...), but PyTorch minimizes loss,
    # so we negate it.
    per_token_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    clipped = clipped_objective < unclipped_objective

    metadata = {
        "ratio": ratio.detach(),
        "clipped_ratio": clipped_ratio.detach(),
        "clipped": clipped.detach(),
        "clip_fraction": clipped.float().mean().detach(),
    }

    return per_token_loss, metadata

def grpo_microbatch_train_step(
    policy_log_probs, response_mask, gradient_accumulation_steps,
    advantages, old_log_probs, cliprange,
):
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    response_mask_float = response_mask.to(per_token_loss.dtype)
    num_response_tokens = response_mask_float.sum(dim=1).clamp(min=1.0)  # (B,)
    per_example_loss = (per_token_loss * response_mask_float).sum(dim=1) / num_response_tokens
    loss = per_example_loss.mean() / gradient_accumulation_steps

    loss.backward()

    metadata = {
        **metadata,
        "loss": loss.detach(),
        "unscaled_loss": loss.detach() * gradient_accumulation_steps,
        "mean_response_length": response_mask_float.sum(dim=1).float().mean().detach(),
    }

    return loss.detach(), metadata
def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    if token_entropies is not None:
        assert len(token_entropies) == len(prompts)

    logs: list[dict[str, Any]] = []

    for i, (prompt, response, ground_truth, reward_info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos)
    ):
        log_entry: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "reward": float(reward_info.get("reward", 0.0)),
            "format_reward": float(reward_info.get("format_reward", 0.0)),
            "answer_reward": float(reward_info.get("answer_reward", 0.0)),
            "response_length_chars": len(response),
            "response_length_words": len(response.split()),
            "reward_info": {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in reward_info.items()
            },
        } 

        if token_entropies is not None:
            log_entry["avg_token_entropy"] = float(token_entropies[i])

        logs.append(log_entry)

    return logs

def train_grpo(
    policy: torch.nn.Module,
    tokenizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    train_prompts: list[str],
    train_ground_truths: list[str],
    val_prompts: list[str] | None = None,
    val_ground_truths: list[str] | None = None,
    n_grpo_steps: int = 8,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    eval_every: int = 5,
    n_val_examples: int = 256,
    max_grad_norm: float = 1.0,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5.

    This version uses HuggingFace generation directly instead of vLLM.
    """

    assert rollout_batch_size % group_size == 0
    assert train_batch_size % gradient_accumulation_steps == 0
    assert train_batch_size <= rollout_batch_size

    if device is None:
        device = next(policy.parameters()).device
    else:
        device = torch.device(device)
        policy.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    logs: dict[str, Any] = {
        "train": [],
        "val": [],
        "generations": [],
    }

    def generate_responses(prompts: list[str]) -> list[str]:
        """Generate responses from policy for a list of prompts."""
        policy.eval()

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        prompt_lengths = encoded["attention_mask"].sum(dim=1)
        stop_token_ids = tokenizer.encode("</answer>", add_special_tokens=False)

        with torch.no_grad():
            generated_ids = policy.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                do_sample=True,
                temperature=sampling_temperature,
                top_p=1.0,
                min_new_tokens=sampling_min_tokens,
                max_new_tokens=sampling_max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id] + stop_token_ids,
            )

        responses: list[str] = []

        for i in range(generated_ids.shape[0]):
            response_ids = generated_ids[i, prompt_lengths[i]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)

        return responses

    def evaluate_validation(step: int) -> dict[str, float]:
        """Evaluate current policy on a small validation subset."""
        if val_prompts is None or val_ground_truths is None:
            return {}

        policy.eval()

        n_eval = min(n_val_examples, len(val_prompts))
        indices = random.sample(range(len(val_prompts)), k=n_eval)

        eval_prompts = [val_prompts[i] for i in indices]
        eval_ground_truths = [val_ground_truths[i] for i in indices]

        eval_responses = generate_responses(eval_prompts)

        reward_infos = [
            reward_fn(response, gt)
            for response, gt in zip(eval_responses, eval_ground_truths)
        ]

        rewards = [
            float(info.get("reward", info.get("total_reward", 0.0)))
            for info in reward_infos
        ]
        format_rewards = [
            float(info.get("format_reward", info.get("format reward", 0.0)))
            for info in reward_infos
        ]
        answer_rewards = [
            float(info.get("answer_reward", info.get("answer reward", 0.0)))
            for info in reward_infos
        ]

        val_log = {
            "step": float(step),
            "val_reward": float(sum(rewards) / len(rewards)),
            "val_format_reward": float(sum(format_rewards) / len(format_rewards)),
            "val_answer_reward": float(sum(answer_rewards) / len(answer_rewards)),
        }

        logs["val"].append(val_log)

        # Save a few readable validation examples.
        generation_logs = log_generations(
            prompts=eval_prompts[:5],
            responses=eval_responses[:5],
            ground_truths=eval_ground_truths[:5],
            reward_infos=reward_infos[:5],
        )
        logs["generations"].append(
            {
                "step": step,
                "examples": generation_logs,
            }
        )

        return val_log

    for step in tqdm(range(1, n_grpo_steps + 1), desc="GRPO steps"):
        policy.eval()

        # ------------------------------------------------------------
        # 1. Sample prompts.
        # ------------------------------------------------------------
        prompt_indices = random.sample(
            range(len(train_prompts)),
            k=n_prompts_per_rollout_batch,
        )

        sampled_prompts = [train_prompts[i] for i in prompt_indices]
        sampled_ground_truths = [train_ground_truths[i] for i in prompt_indices]

        # Repeat each prompt group_size times.
        repeated_prompts: list[str] = []
        repeated_ground_truths: list[str] = []

        for prompt, gt in zip(sampled_prompts, sampled_ground_truths):
            repeated_prompts.extend([prompt] * group_size)
            repeated_ground_truths.extend([gt] * group_size)

        # ------------------------------------------------------------
        # 2. Generate rollout responses from old policy.
        # ------------------------------------------------------------
        rollout_responses = generate_responses(repeated_prompts)

        # ------------------------------------------------------------
        # 3. Compute rewards and group-normalized advantages.
        # ------------------------------------------------------------
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        advantages = advantages.to(device)

        # ------------------------------------------------------------
        # 4. Tokenize prompt + response pairs.
        # ------------------------------------------------------------
        tokenized = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )

        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        # ------------------------------------------------------------
        # 5. Cache old log probs.
        #    Do not differentiate through these.
        # ------------------------------------------------------------
        policy.eval()
        with torch.no_grad():
            old_log_probs = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )["log_probs"]

        old_log_probs = old_log_probs.detach()

        # ------------------------------------------------------------
        # 6. Train on rollout batch.
        # ------------------------------------------------------------
        policy.train()

        indices = list(range(rollout_batch_size))

        total_loss = 0.0
        total_clip_fraction = 0.0
        n_microbatches = 0

        for _epoch in range(epochs_per_rollout_batch):
            random.shuffle(indices)

            for start in range(0, rollout_batch_size, train_batch_size):
                train_indices = indices[start:start + train_batch_size]

                optimizer.zero_grad(set_to_none=True)

                for micro_start in range(0, len(train_indices), micro_train_batch_size):
                    micro_indices = train_indices[
                        micro_start:micro_start + micro_train_batch_size
                    ]

                    mb_input_ids = input_ids[micro_indices]
                    mb_labels = labels[micro_indices]
                    mb_response_mask = response_mask[micro_indices]
                    mb_advantages = advantages[micro_indices]
                    mb_old_log_probs = old_log_probs[micro_indices]

                    policy_outputs = get_response_log_probs(
                        model=policy,
                        input_ids=mb_input_ids,
                        labels=mb_labels,
                        return_token_entropy=False,
                    )

                    policy_log_probs = policy_outputs["log_probs"]

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=mb_response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        advantages=mb_advantages,
                        old_log_probs=mb_old_log_probs,
                        cliprange=cliprange,
                    )

                    total_loss += float(loss.item())
                    total_clip_fraction += float(metadata["clip_fraction"].item())
                    n_microbatches += 1

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    max_grad_norm,
                )

                optimizer.step()

        mean_loss = total_loss / max(n_microbatches, 1)
        mean_clip_fraction = total_clip_fraction / max(n_microbatches, 1)

        train_log = {
            "step": step,
            "loss": mean_loss,
            "clip_fraction": mean_clip_fraction,
            "grad_norm": float(grad_norm.item())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm),
            **reward_metadata,
        }

        logs["train"].append(train_log)

        print(
            f"[step {step}] "
            f"loss={mean_loss:.4f} "
            f"reward={reward_metadata['mean_reward']:.4f} "
            f"answer={reward_metadata['mean_answer_reward']:.4f} "
            f"format={reward_metadata['mean_format_reward']:.4f} "
            f"clip={mean_clip_fraction:.4f}"
        )

        if val_prompts is not None and val_ground_truths is not None:
            if step == 1 or step % eval_every == 0 or step == n_grpo_steps:
                val_log = evaluate_validation(step)
                print(
                    f"  val_reward={val_log.get('val_reward', 0.0):.4f} "
                    f"val_answer={val_log.get('val_answer_reward', 0.0):.4f} "
                    f"val_format={val_log.get('val_format_reward', 0.0):.4f}"
                )

    return logs