import json
import argparse
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from alignment.prompts import DIRECT_PROMPT_TEMPLATE
from alignment.rewards import answer_tag_reward_fn


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return list(dataset)


def build_prompts(
    examples: Sequence[dict[str, Any]],
    prompt_template: str,
) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [
        prompt_template.format(question=example["question"])
        for example in examples
    ]


def truncate_after_answer_tag(response: str) -> str:
    """
    Mimic vLLM stop=['</answer>'] with include_stop_str_in_output=True.
    """
    stop_str = "</answer>"
    if stop_str in response:
        return response.split(stop_str, maxsplit=1)[0] + stop_str
    return response


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[str]:
    """Generate continuations for a batch of prompts using Hugging Face transformers."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[:, prompt_len:]

    responses = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    responses = [truncate_after_answer_tag(r) for r in responses]

    return responses


def evaluate_transformers(
    model,
    tokenizer,
    reward_fn,
    prompts: Sequence[str],
    examples: Sequence[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """
    Generate model outputs, score them, and return serializable evaluation artifacts.
    """
    assert len(prompts) == len(examples)

    results = []

    for start in tqdm(range(0, len(prompts), batch_size)):
        end = min(start + batch_size, len(prompts))

        batch_prompts = list(prompts[start:end])
        batch_examples = list(examples[start:end])

        responses = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        for prompt, example, response in zip(batch_prompts, batch_examples, responses):
            question = example["question"]
            gold_answer = example["answer"]

            reward = reward_fn(response, gold_answer)

            row = {
                "question": question,
                "gold_answer": gold_answer,
                "prompt": prompt,
                "generation": response,
                "format_reward": reward.get("format_reward", 0.0),
                "answer_reward": reward.get("answer_reward", 0.0),
                "reward": reward,
            }

            results.append(row)

    num_examples = len(results)

    format_accuracy = sum(r["format_reward"] for r in results) / num_examples
    answer_accuracy = sum(r["answer_reward"] for r in results) / num_examples

    counts = {
        "format_1_answer_1": sum(
            r["format_reward"] == 1.0 and r["answer_reward"] == 1.0
            for r in results
        ),
        "format_1_answer_0": sum(
            r["format_reward"] == 1.0 and r["answer_reward"] == 0.0
            for r in results
        ),
        "format_0_answer_0": sum(
            r["format_reward"] == 0.0 and r["answer_reward"] == 0.0
            for r in results
        ),
    }

    metrics = {
        "num_examples": num_examples,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,
        "counts": counts,
    }

    return {
        "metrics": metrics,
        "results": results,
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")


def run_direct_baseline(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 2,
    max_examples: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    examples = load_gsm8k_examples(split="test")

    if max_examples is not None:
        examples = examples[:max_examples]

    prompts = build_prompts(
        examples=examples,
        prompt_template=DIRECT_PROMPT_TEMPLATE,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = evaluate_transformers(
        model=model,
        tokenizer=tokenizer,
        reward_fn=answer_tag_reward_fn,
        prompts=prompts,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    results["model_name"] = model_name
    results["prompt_type"] = "direct"
    results["generation_config"] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": "</answer>",
    }

    write_evaluation_results(results, output_path)

    print("\nDirect baseline evaluation complete.")
    print(f"Num examples: {results['metrics']['num_examples']}")
    print(f"Format accuracy: {results['metrics']['format_accuracy']:.4f}")
    print(f"Answer accuracy: {results['metrics']['answer_accuracy']:.4f}")
    print("Counts:")
    print(json.dumps(results["metrics"]["counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_path", type=Path, default=Path("gsm8k_direct_results.json"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    run_direct_baseline(
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
