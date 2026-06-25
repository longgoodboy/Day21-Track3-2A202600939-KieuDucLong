import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs = []
    for rank in [8, 16, 64]:
        path = results_dir / f"train_log_r{rank}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "loss" in frame.columns and "step" in frame.columns:
                curve = frame.loc[frame["loss"].notna(), ["step", "loss"]].copy()
                curve["rank"] = rank
                logs.append(curve)

    if not logs:
        raise FileNotFoundError("No train_log_r*.csv files with loss/step columns found.")

    history = pd.concat(logs, ignore_index=True)
    history.to_csv(results_dir / "loss_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    for rank, frame in history.groupby("rank"):
        plt.plot(frame["step"], frame["loss"], label=f"r={rank}")
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("QLoRA loss curves by LoRA rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=200)
    print(f"Saved {(results_dir / 'loss_history.csv')} and {(results_dir / 'loss_curve.png')}")


if __name__ == "__main__":
    main()
