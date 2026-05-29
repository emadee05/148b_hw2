import json
import argparse
from pathlib import Path
from typing import Any, Sequence
from collections import Counter

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from alignment.prompts import COT_PROMPT_TEMPLATE
from alignment.rewards import (
    answer_tag_reward_fn,
    extract_answer_from_tags,
)
from alignment.drgrpo_grader import grade


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return list(dataset)


def build_prompts(
    examples: Sequence[dict[str, Any]],
    prompt_template: str,
) -> list[str]:
    return [
        prompt_template.format(question=example["question"])
        for example in examples
    ]


def truncate_after_answer_tag(response: str) -> str:
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
    num_return_sequences: int = 1,
) -> list[str]:
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
            num_return_sequences=num_return_sequences,
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


def majority_vote_with_counts(responses: list[str]):
    """
    Use the provided tag extractor, but also return counts/tie info
    for the writeup.
    """
    answers = [
        extract_answer_from_tags(response)
        for response in responses
    ]
    valid_answers = [a for a in answers if a is not None]

    if len(valid_answers) == 0:
        return None, answers, {}, True

    counts = Counter(valid_answers)
    max_count = max(counts.values())
    winners = [ans for ans, count in counts.items() if count == max_count]
    is_tie = len(winners) > 1

    # deterministic tie-break: choose first tied answer that appeared
    for ans in valid_answers:
        if ans in winners:
            majority_answer = ans
            break

    return majority_answer, answers, dict(counts), is_tie


def evaluate_cot(
    model,
    tokenizer,
    prompts: Sequence[str],
    examples: Sequence[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
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
            num_return_sequences=1,
        )

        for prompt, example, response in zip(batch_prompts, batch_examples, responses):
            gold_answer = example["answer"]
            reward = answer_tag_reward_fn(response, gold_answer)
            extracted_answer = extract_answer_from_tags(response)

            results.append(
                {
                    "question": example["question"],
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "generation": response,
                    "extracted_answer": extracted_answer,
                    "format_reward": reward.get("format_reward", 0.0),
                    "answer_reward": reward.get("answer_reward", 0.0),
                    "reward": reward,
                }
            )

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

    return {
        "mode": "cot",
        "metrics": {
            "num_examples": num_examples,
            "format_accuracy": format_accuracy,
            "answer_accuracy": answer_accuracy,
            "counts": counts,
        },
        "results": results,
    }


def evaluate_self_consistency(
    model,
    tokenizer,
    prompts: Sequence[str],
    examples: Sequence[dict[str, Any]],
    batch_size: int,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    results = []

    for start in tqdm(range(0, len(prompts), batch_size)):
        end = min(start + batch_size, len(prompts))

        batch_prompts = list(prompts[start:end])
        batch_examples = list(examples[start:end])

        # This returns batch_size * k responses ordered as:
        # prompt0 sample0, prompt0 sample1, ..., prompt1 sample0, ...
        flat_responses = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
        )

        grouped_responses = [
            flat_responses[i : i + k]
            for i in range(0, len(flat_responses), k)
        ]

        for prompt, example, responses in zip(batch_prompts, batch_examples, grouped_responses):
            gold_answer = example["answer"]

            sample_rows = []
            for response in responses:
                extracted = extract_answer_from_tags(response)
                reward = answer_tag_reward_fn(response, gold_answer)
                sample_rows.append(
                    {
                        "generation": response,
                        "extracted_answer": extracted,
                        "reward": reward,
                    }
                )

            majority_answer, extracted_answers, answer_counts, is_tie = majority_vote_with_counts(responses)

            if majority_answer is None:
                majority_reward = {
                    "format_reward": 0.0,
                    "answer_reward": 0.0,
                    "reward": 0.0,
                }
            else:
                is_correct = float(grade(majority_answer, gold_answer))
                majority_reward = {
                    "format_reward": 1.0,
                    "answer_reward": is_correct,
                    "reward": is_correct,
                }

            results.append(
                {
                    "question": example["question"],
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "generations": sample_rows,
                    "extracted_answers": extracted_answers,
                    "answer_counts": answer_counts,
                    "majority_answer": majority_answer,
                    "tie": is_tie,
                    "majority_reward": majority_reward,
                }
            )

    num_examples = len(results)
    majority_format_accuracy = (
        sum(r["majority_reward"]["format_reward"] for r in results) / num_examples
    )
    majority_answer_accuracy = (
        sum(r["majority_reward"]["answer_reward"] for r in results) / num_examples
    )
    tie_rate = sum(1 for r in results if r["tie"]) / num_examples
    avg_unique_answers = sum(len(r["answer_counts"]) for r in results) / num_examples

    return {
        "mode": "self_consistency",
        "k": k,
        "metrics": {
            "num_examples": num_examples,
            "k": k,
            "majority_format_accuracy": majority_format_accuracy,
            "majority_answer_accuracy": majority_answer_accuracy,
            "tie_rate": tie_rate,
            "avg_unique_answers_per_question": avg_unique_answers,
        },
        "results": results,
    }


def load_model_and_tokenizer(model_name: str):
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
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["cot", "self_consistency"], required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    examples = load_gsm8k_examples(split="test")

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    prompts = build_prompts(
        examples=examples,
        prompt_template=str(COT_PROMPT_TEMPLATE),
    )

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    if args.mode == "cot":
        output = evaluate_cot(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            examples=examples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        output = evaluate_self_consistency(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            examples=examples,
            batch_size=args.batch_size,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    output["model_name"] = args.model_name
    output["generation_config"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stop": "</answer>",
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to: {args.output_path}")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
