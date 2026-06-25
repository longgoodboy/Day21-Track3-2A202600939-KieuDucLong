import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_ADAPTERS_DIR = Path("adapters")
DEFAULT_DATA_DIR = Path("data/processed")

TEST_PROMPTS = [
    "Giải thích QLoRA là gì theo cách dễ hiểu cho sinh viên năm nhất.",
    "Viết hướng dẫn từng bước để chuẩn bị dataset Alpaca format cho fine-tuning.",
    "So sánh ngắn gọn giữa prompt engineering, RAG và fine-tuning bằng tiếng Việt.",
    "Trả lời theo định dạng gạch đầu dòng: khi nào nên dùng LoRA rank thấp?",
    "Hãy tóm tắt đoạn sau thành 3 câu lịch sự: Fine-tuning giúp mô hình học style và format tốt hơn.",
    "Nếu không chắc câu trả lời, hãy nói rõ giới hạn thay vì đoán. Ví dụ: rank cao hơn có luôn tốt hơn không?",
    "Viết một lời giải thích thân thiện cho học sinh về perplexity trong đánh giá mô hình ngôn ngữ.",
    "Cho một ví dụ ngắn về input và output trong dataset Alpaca.",
    "Giải thích vì sao phải giữ experiment fair khi so sánh r=8, r=16, r=64.",
    "Viết một câu trả lời dạng tutoring style cho câu hỏi: max_seq_length nên chọn như thế nào?",
]


def read_dataset_stats(data_dir: Path) -> dict:
    stats_path = data_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run scripts/prepare_dataset.py first."
        )
    return json.loads(stats_path.read_text(encoding="utf-8"))


def load_eval_dataset(data_dir: Path):
    return load_dataset("json", data_files=str(data_dir / "eval.jsonl"), split="train")


def load_base_model(max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def eval_loss_for_model(model, tokenizer, eval_ds, max_seq_length: int) -> float:
    model.eval()
    losses = []
    for row in eval_ds:
        encoded = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
        losses.append(float(outputs.loss.detach().cpu()))
    if not losses:
        raise ValueError("Evaluation dataset is empty.")
    return float(sum(losses) / len(losses))


def generation_prompt(prompt: str) -> str:
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    FastLanguageModel.for_inference(model)
    encoded = tokenizer(generation_prompt(prompt), return_tensors="pt").to(model.device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--adapters-dir", default=str(DEFAULT_ADAPTERS_DIR))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation script.")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    adapters_dir = Path(args.adapters_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = read_dataset_stats(data_dir)
    max_seq_length = int(dataset_stats["token_length"]["recommended_max_seq_length"])
    eval_ds = load_eval_dataset(data_dir)

    summary_csv = results_dir / "rank_experiment_summary.csv"
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame(
            columns=[
                "rank",
                "lora_alpha",
                "target_modules",
                "trainable_params",
                "trainable_percent",
                "training_time_min",
                "peak_vram_gb",
                "eval_loss",
                "perplexity",
            ]
        )

    base_model, tokenizer = load_base_model(max_seq_length=max_seq_length)
    base_eval_loss = eval_loss_for_model(base_model, tokenizer, eval_ds, max_seq_length=max_seq_length)
    base_row = {
        "rank": "base",
        "lora_alpha": "-",
        "target_modules": "-",
        "trainable_params": 0,
        "trainable_percent": 0,
        "training_time_min": "-",
        "peak_vram_gb": "-",
        "eval_loss": base_eval_loss,
        "perplexity": math.exp(base_eval_loss),
    }

    summary = summary.loc[summary["rank"].astype(str) != "base"].copy()
    summary = pd.concat([summary, pd.DataFrame([base_row])], ignore_index=True)
    summary.to_csv(summary_csv, index=False)

    outputs = {"base": base_model}
    for rank in [8, 16, 64]:
        adapter_path = adapters_dir / f"r{rank}"
        if adapter_path.exists():
            fresh_base, _ = load_base_model(max_seq_length=max_seq_length)
            outputs[f"r{rank}"] = PeftModel.from_pretrained(fresh_base, str(adapter_path))

    rows = []
    for prompt in TEST_PROMPTS:
        row = {"prompt": prompt}
        for key, model in outputs.items():
            row[f"{key}_output"] = generate_response(model, tokenizer, prompt)
        available_ranks = [rank for rank in [8, 16, 64] if f"r{rank}_output" in row]
        if available_ranks:
            best_rank = min(
                available_ranks,
                key=lambda rank: float(
                    summary.loc[summary["rank"].astype(str) == str(rank), "perplexity"].iloc[0]
                ),
            )
            row["best_rank"] = best_rank
        else:
            row["best_rank"] = ""
        row["comment"] = "TODO: add human qualitative judgment after reviewing generated outputs."
        rows.append(row)

    pd.DataFrame(rows).to_csv(results_dir / "qualitative_comparison.csv", index=False)
    print(json.dumps({"summary_csv": str(summary_csv), "qualitative_csv": str(results_dir / "qualitative_comparison.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
