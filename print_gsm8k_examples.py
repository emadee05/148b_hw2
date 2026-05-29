import json
import argparse
from pathlib import Path

from alignment.rewards import extract_answer_from_tags
from alignment.drgrpo_grader import grade


def print_direct_or_cot_examples(data: dict, num_examples: int) -> None:
    results = data["results"]

    for i, row in enumerate(results[:num_examples]):
        generation = row.get("generation", "")
        extracted = row.get("extracted_answer", None)

        if extracted is None:
            extracted = extract_answer_from_tags(generation)

        print("=" * 100)
        print(f"Example {i}")
        print("QUESTION:")
        print(row.get("question"))

        print("\nGOLD ANSWER:")
        print(row.get("gold_answer"))

        print("\nGENERATION:")
        print(generation)

        print("\nEXTRACTED ANSWER:")
        print(repr(extracted))

        print("\nREWARD:")
        if "reward" in row:
            print(row["reward"])
        else:
            print(
                {
                    "format_reward": row.get("format_reward"),
                    "answer_reward": row.get("answer_reward"),
                }
            )

        if extracted is not None:
            print("\nMANUAL GRADE CHECK:")
            print(grade(extracted, row["gold_answer"]))


def print_self_consistency_examples(data: dict, num_examples: int) -> None:
    results = data["results"]

    for i, row in enumerate(results[:num_examples]):
        print("=" * 100)
        print(f"Example {i}")
        print("QUESTION:")
        print(row.get("question"))

        print("\nGOLD ANSWER:")
        print(row.get("gold_answer"))

        print("\nEXTRACTED ANSWERS:")
        print(row.get("extracted_answers"))

        print("\nANSWER COUNTS:")
        print(row.get("answer_counts"))

        print("\nMAJORITY ANSWER:")
        print(repr(row.get("majority_answer")))

        print("\nTIE:")
        print(row.get("tie"))

        print("\nMAJORITY REWARD:")
        print(row.get("majority_reward"))

        majority_answer = row.get("majority_answer")
        if majority_answer is not None:
            print("\nMANUAL MAJORITY GRADE CHECK:")
            print(grade(majority_answer, row["gold_answer"]))

        print("\nGENERATIONS:")
        for j, sample in enumerate(row.get("generations", [])):
            print("-" * 80)
            print(f"Sample {j}")
            print("Extracted:", repr(sample.get("extracted_answer")))
            print("Reward:", sample.get("reward"))
            print(sample.get("generation"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        data = json.load(f)

    mode = data.get("mode", "")

    print(f"Loaded: {args.input_path}")
    print("Metrics:")
    print(json.dumps(data.get("metrics", {}), indent=2))
    print()

    if mode == "self_consistency":
        print_self_consistency_examples(data, args.num_examples)
    else:
        print_direct_or_cot_examples(data, args.num_examples)


if __name__ == "__main__":
    main()
