import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


DEFAULT_DATASET = "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        normalized = column.lower().replace("_vi", "").replace("_en", "")
        if normalized in candidates:
            return column
    return None


def resolve_columns(dataset: Dataset) -> tuple[str, str | None, str]:
    columns = list(dataset.column_names)
    instruction_col = choose_column(columns, ["instruction", "prompt", "question"])
    input_col = choose_column(columns, ["input", "context"])
    output_col = choose_column(columns, ["output", "response", "answer", "completion"])
    if instruction_col is None or output_col is None:
        raise ValueError(
            f"Could not infer Alpaca columns from dataset columns: {columns}"
        )
    return instruction_col, input_col, output_col


def format_prompt(instruction: str, input_text: str, output: str = "") -> str:
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def build_dataframe(dataset: Dataset) -> pd.DataFrame:
    instruction_col, input_col, output_col = resolve_columns(dataset)
    frame = dataset.to_pandas()
    frame["instruction"] = frame[instruction_col].map(normalize_text)
    frame["input"] = (
        frame[input_col].map(normalize_text) if input_col is not None else ""
    )
    frame["output"] = frame[output_col].map(normalize_text)
    frame = frame[["instruction", "input", "output"]].copy()
    return frame


def clean_dataframe(frame: pd.DataFrame, min_output_words: int) -> tuple[pd.DataFrame, dict]:
    original_rows = len(frame)
    frame = frame.dropna(subset=["instruction", "output"]).copy()
    frame["instruction"] = frame["instruction"].astype(str).str.strip()
    frame["input"] = frame["input"].astype(str).str.strip()
    frame["output"] = frame["output"].astype(str).str.strip()

    non_empty_mask = (frame["instruction"] != "") & (frame["output"] != "")
    frame = frame.loc[non_empty_mask].copy()

    frame["output_word_count"] = frame["output"].str.split().str.len()
    frame = frame.loc[frame["output_word_count"] >= min_output_words].copy()

    frame["dedup_key"] = (
        frame["instruction"] + "\n<SEP>\n" + frame["input"] + "\n<SEP>\n" + frame["output"]
    )
    frame = frame.drop_duplicates(subset=["dedup_key"]).copy()
    frame["text"] = frame.apply(
        lambda row: format_prompt(row["instruction"], row["input"], row["output"]),
        axis=1,
    )

    cleanup = {
        "original_rows": int(original_rows),
        "clean_rows": int(len(frame)),
        "removed_rows": int(original_rows - len(frame)),
    }
    return frame.drop(columns=["output_word_count", "dedup_key"]), cleanup


def compute_length_stats(texts: list[str], tokenizer_name: str) -> tuple[dict, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]
    p95 = int(np.percentile(lengths, 95))
    chosen = min(1024, 1 << (max(p95, 256) - 1).bit_length())
    stats = {
        "tokenizer": tokenizer_name,
        "min": int(min(lengths)),
        "mean": float(np.mean(lengths)),
        "median": int(median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": p95,
        "p99": int(np.percentile(lengths, 99)),
        "max": int(max(lengths)),
        "recommended_max_seq_length": int(chosen),
    }
    return stats, lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-output-words", type=int, default=8)
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    if args.sample_size and len(dataset) > args.sample_size:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.sample_size))

    frame = build_dataframe(dataset)
    cleaned_frame, cleanup = clean_dataframe(frame, args.min_output_words)
    dataset_clean = Dataset.from_pandas(cleaned_frame, preserve_index=False)
    split = dataset_clean.train_test_split(test_size=0.1, seed=args.seed)

    token_stats, _ = compute_length_stats(cleaned_frame["text"].tolist(), args.tokenizer)

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    all_path = output_dir / "dataset_full.jsonl"
    stats_path = output_dir / "dataset_stats.json"

    split["train"].to_json(str(train_path), force_ascii=False)
    split["test"].to_json(str(eval_path), force_ascii=False)
    dataset_clean.to_json(str(all_path), force_ascii=False)

    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "train_size": int(len(split["train"])),
        "eval_size": int(len(split["test"])),
        "cleanup": cleanup,
        "token_length": token_stats,
        "prompt_template": {
            "with_input": "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:\\n{output}",
            "without_input": "### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}",
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "stats_path": str(stats_path),
        "train_size": stats["train_size"],
        "eval_size": stats["eval_size"],
        "recommended_max_seq_length": token_stats["recommended_max_seq_length"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
