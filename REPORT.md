# Lab 21 - LoRA / QLoRA Fine-tuning Report

## 1. Setup

- Base model: `unsloth/Qwen2.5-3B-bnb-4bit` on the training notebook/scripts.
- Quantization: QLoRA 4-bit via Unsloth.
- LoRA target modules: `q_proj`, `v_proj`.
- Shared hyperparameters across fair runs: 3 epochs, learning rate `2e-4`, cosine scheduler, warmup ratio `0.10`, `optim="adamw_8bit"`, gradient checkpointing enabled, `seed=42`.
- Shared batching across fair runs: `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, effective batch size `8`.
- Dataset source: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, converted/cleaned into Alpaca fields.
- Colab training environment used for actual experiments: Tesla T4 16 GB.
- Colab install note: use `bash scripts/install_colab.sh` instead of `pip install -r requirements.txt` because Unsloth and TRL need notebook-style installation order on Colab.
- Clean dataset size: 284 examples after filtering from 300 sampled examples.
- Train/eval split: 255 train / 29 eval with seed `42`.
- Token length stats: min `28`, mean `244.04`, median `202`, p90 `528`, p95 `563`, p99 `708`, max `738`.
- Chosen `max_seq_length`: `1024`, following the p95-based rounding rule while staying within the T4-safe cap.

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

Actual preprocessing results from `data/processed/dataset_stats.json`:

- 300 samples were drawn from the source split.
- 16 rows were removed during cleaning.
- 284 rows remained after normalization, deduplication, and minimum-output filtering.
- The fixed split produced 255 training examples and 29 evaluation examples.
- The p95 token length was 563, so `max_seq_length` was rounded up to 1024 to cover almost all examples while keeping the setting T4-safe.

## 3. Rank Experiment Results

The repository now includes [`scripts/train_lora_rank.py`](/D:/vin_lab/Day21-Track3-2A202600939-KieuDucLong/scripts/train_lora_rank.py), which trains one rank at a time while keeping all other settings fixed. Only `rank` and `lora_alpha = 2 * rank` change across runs:

| Experiment | Rank | Alpha |
|---|---:|---:|
| low rank | 8 | 16 |
| baseline | 16 | 32 |
| high rank | 64 | 128 |

Actual output file: `results/rank_experiment_summary.csv`

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---|---:|---:|---:|---:|---:|
| 8 | 1,843,200 | 5.95 min | 10.27 GB | 1.5160 | 4.5542 |
| 16 | 3,686,400 | 6.00 min | 10.29 GB | 1.4691 | 4.3455 |
| 64 | 14,745,600 | 5.98 min | 10.43 GB | 1.4629 | 4.3185 |

Observations:

- `r=64` achieved the lowest eval loss and lowest perplexity.
- `r=8` used the fewest trainable parameters but also had the weakest perplexity.
- Training time was almost identical across all three runs on the sampled dataset, so the main cost difference came from trainable parameter count and slightly higher VRAM for larger rank.
- The practical gap between `r=16` and `r=64` was small: perplexity improved from `4.3455` to `4.3185`, while trainable parameter count increased by 4x.

Note:

- The base-model row is not included here because the pasted Colab log captured the adapter runs clearly but did not include the final base-model evaluation row from `evaluate_adapters.py`. I am not inventing that missing number.

## 4. Loss Curve Analysis

The repository now includes [`scripts/plot_results.py`](/D:/vin_lab/Day21-Track3-2A202600939-KieuDucLong/scripts/plot_results.py), which merges per-rank trainer logs into:

- `results/loss_history.csv`
- `results/loss_curve.png`

The loss curves were generated successfully on Colab into `results/loss_curve.png`, and the per-rank log histories were exported into `results/loss_history.csv`.

Based on the Colab logs:

- All three runs showed a generally smooth downward trend in training loss across 96 steps.
- `r=16` decreased from early losses around `1.65-1.73` to late losses around `1.30-1.41`.
- `r=8` followed a similar shape but converged to slightly worse evaluation metrics.
- `r=64` showed the strongest late-stage training loss values and the best final perplexity, but the gain over `r=16` was modest rather than dramatic.
- There was no obvious sign of instability or divergence in any run.

Because the actual image file was generated in Colab and not copied back into this local workspace, the analysis here is based on the recorded logs rather than an embedded local image preview.

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

Status note:

- The Colab log confirms that `results/qualitative_comparison.csv` was generated successfully.
- However, the actual generated text outputs were not included in the pasted log available in this workspace, so I cannot truthfully summarize those responses here without the file contents.

## 6. Conclusion: Rank Trade-off

Among the three fair QLoRA runs, `r=64` achieved the best quantitative result with the lowest eval loss (`1.4629`) and lowest perplexity (`4.3185`). However, the gain over `r=16` was very small: `r=16` reached perplexity `4.3455`, which is extremely close, while using only one quarter of the trainable parameters of `r=64`. Peak VRAM also increased only slightly from about `10.29 GB` to `10.43 GB`, and training time stayed near 6 minutes for all three runs because the dataset was small. This means the main trade-off is not runtime here, but parameter efficiency and model simplicity.

If GPU memory is limited or a compact adapter is preferred, `r=8` is the safest choice, but it gives the weakest perplexity of the three runs. If quality is the only priority, `r=64` is the best by the recorded metrics. For an overall production-style recommendation in this lab setting, I would choose `r=16` as the best trade-off. It is much stronger than `r=8`, nearly matches `r=64`, and avoids the 4x increase in trainable parameters that came with only a marginal perplexity improvement. In other words, `r=64` improved quality, but not by enough to clearly justify its extra adapter capacity for this dataset size.

## 7. What I Learned

- LoRA rank controls adapter capacity, so increasing rank can improve expressiveness but also increases trainable parameters and memory cost.
- QLoRA makes larger instruction models trainable on limited GPU hardware by quantizing the frozen base model to 4-bit.
- Perplexity is useful for comparison, but it is not enough on its own; qualitative inspection is needed to verify instruction-following style and formatting behavior.

## 8. Limitations and Future Work

- This local workspace did not have CUDA, so the training artifacts had to be produced in Colab and then interpreted from exported logs.
- The current training target modules follow the core rubric (`q_proj`, `v_proj`) rather than the broader all-layers stretch configuration.
- The base-model eval row and the full qualitative generations were produced in Colab but were not copied back into this workspace, so they are not restated here without source outputs.
- Future work after core completion:
  - push the best adapter to Hugging Face Hub,
  - try all-target-modules LoRA,
  - add W&B tracking for cleaner experiment history.
