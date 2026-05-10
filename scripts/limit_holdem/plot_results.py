import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, x, y, output_path, title):
    plt.figure()
    plt.plot(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    df = pd.read_csv(metrics_path)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = metrics_path.stem

    if "avg_payoff" in df.columns:
        plot_metric(
            df,
            "episode",
            "avg_payoff",
            out_dir / f"{stem}_payoff.png",
            "DQN Evaluation Payoff Over Training",
        )

    if "avg_loss" in df.columns:
        plot_metric(
            df,
            "episode",
            "avg_loss",
            out_dir / f"{stem}_loss.png",
            "DQN Loss Over Training",
        )

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()