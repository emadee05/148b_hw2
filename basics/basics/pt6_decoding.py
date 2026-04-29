from pt2_tokenizer import Tokenizer
from pt3_linear import TransformerLM, softmax
from pt4_crossentropy import cross_entropy
from pt5_training import data_loading


import torch

def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return softmax(logits / temperature, dim=-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = cumulative_probs <= p
    keep_mask[..., 0] = True  # always keep at least the top token

    filtered_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_indices.gather(dim=-1, index=sampled_sorted_idx)
    return next_token


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None) -> torch.Tensor:
    # logits: (B, vocab_size)

    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = softmax_with_temperature(logits, temperature)

    if top_p is not None and top_p < 1.0:
        return sample_top_p(probs, top_p)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_id: int | None = None) -> torch.Tensor:
    # prompt_tokens: (B, T)
    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        # optionally crop to model context length
        idx_cond = tokens[:, -model.context_length:]

        logits = model(idx_cond)          # (B, T, vocab_size)
        next_logits = logits[:, -1, :]    # (B, vocab_size)

        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_id is not None:
            if torch.all(next_token.squeeze(-1) == eos_id):
                break

    return tokens

