import argparse
import gc
import inspect
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_ADAPTERS_DIR = Path("adapters")
DEFAULT_DATA_DIR = Path("data/processed")


def read_dataset_stats(data_dir: Path) -> dict:
    stats_path = data_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run scripts/prepare_dataset.py first."
        )
    return json.loads(stats_path.read_text(encoding="utf-8"))


def load_splits(data_dir: Path):
    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    eval_ds = load_dataset("json", data_files=str(data_dir / "eval.jsonl"), split="train")
    return train_ds, eval_ds


def load_base_model(max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def wrap_with_lora(model, rank: int, alpha: int):
    return FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )


def patch_trainer_alias() -> None:
    import unsloth.models._utils as utils_module

    underlying_init = getattr(utils_module, "_original_trainer_init", Trainer.__init__)
    if getattr(underlying_init, "_aliased", False):
        return

    def aliased_trainer_init(self, *args, **kwargs):
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        return underlying_init(self, *args, **kwargs)

    aliased_trainer_init._aliased = True
    utils_module._original_trainer_init = aliased_trainer_init

    if "tokenizer" not in inspect.signature(Trainer.__init__).parameters:
        original_init = Trainer.__init__

        def trainer_init(self, *args, **kwargs):
            if "tokenizer" in kwargs and "processing_class" not in kwargs:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
            return original_init(self, *args, **kwargs)

        trainer_init._aliased = True
        Trainer.__init__ = trainer_init


def make_trainer(model, tokenizer, train_ds, eval_ds, output_dir: str, max_seq_length: int):
    patch_trainer_alias()

    try:
        from trl import SFTConfig

        has_sft_config = True
    except ImportError:
        SFTConfig = None
        has_sft_config = False

    training_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        prediction_loss_only=True,
        warmup_ratio=0.10,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    training_args[eval_key] = "no"

    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    supports_old_kwargs = "dataset_text_field" in sft_params

    if has_sft_config:
        sft_params_config = inspect.signature(SFTConfig.__init__).parameters
        extra = {
            "dataset_text_field": "text",
            "packing": False,
            "max_seq_length": max_seq_length,
        }
        extra = {key: value for key, value in extra.items() if key in sft_params_config}
        valid_training_args = {
            key: value for key, value in training_args.items() if key in sft_params_config
        }
        args = SFTConfig(**valid_training_args, **extra)
    else:
        args = TrainingArguments(**training_args)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": args,
    }
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if supports_old_kwargs:
        trainer_kwargs.update(
            {"dataset_text_field": "text", "max_seq_length": max_seq_length, "packing": False}
        )
    return SFTTrainer(**trainer_kwargs)


def safe_evaluate(trainer) -> float:
    try:
        metrics = trainer.evaluate()
        return float(metrics["eval_loss"])
    except Exception:
        model = trainer.model
        model.eval()
        losses = []
        for row in trainer.eval_dataset:
            tokenized = trainer.processing_class(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=getattr(trainer.args, "max_seq_length", None),
            )
            tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
            with torch.no_grad():
                outputs = model(**tokenized, labels=tokenized["input_ids"])
            losses.append(float(outputs.loss.detach().cpu()))
        if not losses:
            raise ValueError("Evaluation produced no losses.")
        return float(sum(losses) / len(losses))


def append_metrics(row: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        frame = pd.read_csv(output_csv)
        frame = frame.loc[frame["rank"] != row["rank"]].copy()
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("rank").reset_index(drop=True)
    frame.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, choices=[8, 16, 64])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--adapters-dir", default=str(DEFAULT_ADAPTERS_DIR))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script.")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    adapters_dir = Path(args.adapters_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    alpha = args.rank * 2
    dataset_stats = read_dataset_stats(data_dir)
    max_seq_length = int(dataset_stats["token_length"]["recommended_max_seq_length"])
    train_ds, eval_ds = load_splits(data_dir)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_base_model(max_seq_length=max_seq_length)
    model = wrap_with_lora(model, rank=args.rank, alpha=alpha)
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=str(results_dir / f"trainer_r{args.rank}"),
        max_seq_length=max_seq_length,
    )

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())

    start_time = time.time()
    trainer.train()
    training_time_min = (time.time() - start_time) / 60
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    adapter_dir = adapters_dir / f"r{args.rank}"
    trainer.save_model(str(adapter_dir))

    log_history = pd.DataFrame(trainer.state.log_history)
    log_history["rank"] = args.rank
    log_path = results_dir / f"train_log_r{args.rank}.csv"
    log_history.to_csv(log_path, index=False)

    eval_loss = safe_evaluate(trainer)
    perplexity = math.exp(eval_loss)

    row = {
        "rank": args.rank,
        "lora_alpha": alpha,
        "target_modules": "q_proj,v_proj",
        "trainable_params": int(trainable_params),
        "trainable_percent": 100 * trainable_params / total_params,
        "training_time_min": training_time_min,
        "peak_vram_gb": peak_vram_gb,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "log_path": str(log_path).replace("\\", "/"),
        "adapter_path": str(adapter_dir).replace("\\", "/"),
    }

    append_metrics(row, results_dir / "rank_experiment_summary.csv")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
