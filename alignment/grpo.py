from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


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
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Only response tokens should contribute.
    response_mask = response_mask.to(per_token_loss.dtype)

    # Average loss over response tokens for each example.
    # Shape: (B,)
    per_example_loss = masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        normalize_constant=response_mask.sum(dim=1).clamp(min=1.0),
        dim=1,
    )

    # Average across examples in the microbatch.
    loss = per_example_loss.mean()

    # Scale for gradient accumulation.
    loss = loss / gradient_accumulation_steps

    # Backprop happens inside this function, per the assignment.
    loss.backward()

    metadata = {
        **metadata,
        "loss": loss.detach(),
        "unscaled_loss": (loss.detach() * gradient_accumulation_steps),
        "mean_response_length": response_mask.sum(dim=1).float().mean().detach(),
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

def train_grpo(*args, **kwargs) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    raise NotImplementedError
