# Lab 21 - LoRA / QLoRA Fine-tuning Report

## 1. Setup

- Base model: `unsloth/Qwen2.5-3B-bnb-4bit` on the training notebook/scripts.
- Quantization: QLoRA 4-bit via Unsloth.
- LoRA target modules: `q_proj`, `v_proj`.
- Shared hyperparameters across fair runs: 3 epochs, learning rate `2e-4`, cosine scheduler, warmup ratio `0.10`, `optim="adamw_8bit"`, gradient checkpointing enabled, `seed=42`.
- Shared batching across fair runs: `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, effective batch size `8`.
- Dataset source plan: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, converted/cleaned into Alpaca fields.
- GPU status in this local workspace on June 25, 2026: no CUDA GPU available (`torch 2.8.0+cpu`), so QLoRA training/evaluation metrics have not been generated here.
- Colab install note: use `bash scripts/install_colab.sh` instead of `pip install -r requirements.txt` because Unsloth and TRL need notebook-style installation order on Colab.

TODO:
- Run the training pipeline on Colab T4 (or equivalent CUDA GPU).
- Fill in actual GPU name, VRAM, runtime cost estimate, dataset size after cleaning, and chosen `max_seq_length` from `data/processed/dataset_stats.json`.

## 2. Dataset Preparation

The repository now includes [`scripts/prepare_dataset.py`](/D:/vin_lab/Day21-Track3-2A202600939-KieuDucLong/scripts/prepare_dataset.py), which:

- loads the dataset,
- normalizes it to Alpaca fields `instruction`, `input`, `output`,
- removes empty/duplicate/too-short examples,
- creates a `text` column with the training prompt template,
- performs a fixed-seed `90/10` train/eval split,
- computes token-length statistics including `p90`, `p95`, and recommended `max_seq_length`,
- saves `train.jsonl`, `eval.jsonl`, `dataset_full.jsonl`, and `dataset_stats.json`.

Prompt template used for training:

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

When `input` is empty, the template becomes:

```text
### Instruction:
{instruction}

### Response:
{output}
```

TODO:
- Run `python scripts/prepare_dataset.py --sample-size 300`.
- Copy actual cleanup counts, train/eval sizes, and token-length statistics into this section.
- State the final `p95` and explain the chosen `max_seq_length`.

## 3. Rank Experiment Results

The repository now includes [`scripts/train_lora_rank.py`](/D:/vin_lab/Day21-Track3-2A202600939-KieuDucLong/scripts/train_lora_rank.py), which trains one rank at a time while keeping all other settings fixed. Only `rank` and `lora_alpha = 2 * rank` change across runs:

| Experiment | Rank | Alpha |
|---|---:|---:|
| low rank | 8 | 16 |
| baseline | 16 | 32 |
| high rank | 64 | 128 |

Expected output file: `results/rank_experiment_summary.csv`

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---|---:|---:|---:|---:|---:|
| base | TODO | TODO | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO | TODO | TODO |
| 64 | TODO | TODO | TODO | TODO | TODO |

TODO:
- Run the three fair training jobs on GPU.
- Run adapter/base evaluation to fill real `eval_loss` and `perplexity`.
- Replace all placeholders with actual metrics from `results/rank_experiment_summary.csv`.

## 4. Loss Curve Analysis

The repository now includes [`scripts/plot_results.py`](/D:/vin_lab/Day21-Track3-2A202600939-KieuDucLong/scripts/plot_results.py), which merges per-rank trainer logs into:

- `results/loss_history.csv`
- `results/loss_curve.png`

Planned analysis points:

- whether loss decreases smoothly for each rank,
- whether any run looks unstable,
- whether `r=64` shows signs of overfitting or only marginal gain,
- whether `r=16` is the best trade-off.

TODO:
- Run `python scripts/plot_results.py` after all three training runs complete.
- Insert the produced image here and add observations based on the real curve.

## 5. Qualitative Comparison

The repository now includes [`scripts/evaluate_adapters.py`](/D:/vin_lab/Day21-Track3-2A202600939-KieuDucLong/scripts/evaluate_adapters.py), which:

- evaluates the base model on the same eval set,
- reloads adapters `r=8`, `r=16`, `r=64`,
- generates outputs for 10 prompts,
- saves `results/qualitative_comparison.csv`.

Representative examples to discuss after generation:

1. Prompt: `Giải thích QLoRA là gì theo cách dễ hiểu cho sinh viên năm nhất.`
   TODO: summarize base output, summarize best fine-tuned output, add judgment.
2. Prompt: `Viết hướng dẫn từng bước để chuẩn bị dataset Alpaca format cho fine-tuning.`
   TODO: summarize base output, summarize best fine-tuned output, add judgment.
3. Prompt: `So sánh ngắn gọn giữa prompt engineering, RAG và fine-tuning bằng tiếng Việt.`
   TODO: summarize base output, summarize best fine-tuned output, add judgment.
4. Prompt: `Trả lời theo định dạng gạch đầu dòng: khi nào nên dùng LoRA rank thấp?`
   TODO: summarize base output, summarize best fine-tuned output, add judgment.
5. Prompt: `Giải thích vì sao phải giữ experiment fair khi so sánh r=8, r=16, r=64.`
   TODO: summarize base output, summarize best fine-tuned output, add judgment.

Important note:

- Do not cherry-pick only wins.
- Keep at least one example where fine-tuning is similar to base or not clearly better.

## 6. Conclusion: Rank Trade-off

TODO:
- Write this section after collecting real metrics.
- The final conclusion must answer:
  - Which rank is best overall?
  - Which rank is best under tight GPU memory?
  - Which rank is best if quality is the only priority?
  - Whether `r=64` justifies its extra cost.
  - Which rank should be chosen for production in this lab setting.

Placeholder guidance:

If `r=16` achieves similar perplexity and qualitative quality to `r=64` while using less VRAM and less training time, then `r=16` should be selected as the best practical trade-off.

## 7. What I Learned

- LoRA rank controls adapter capacity, so increasing rank can improve expressiveness but also increases trainable parameters and memory cost.
- QLoRA makes larger instruction models trainable on limited GPU hardware by quantizing the frozen base model to 4-bit.
- Perplexity is useful for comparison, but it is not enough on its own; qualitative inspection is needed to verify instruction-following style and formatting behavior.

## 8. Limitations and Future Work

- This local workspace did not have CUDA, so no training metrics were fabricated; all missing numbers remain explicit TODOs until run on Colab or another GPU environment.
- The current training target modules follow the core rubric (`q_proj`, `v_proj`) rather than the broader all-layers stretch configuration.
- Future work after core completion:
  - push the best adapter to Hugging Face Hub,
  - try all-target-modules LoRA,
  - add W&B tracking for cleaner experiment history.
