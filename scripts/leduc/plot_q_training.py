import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_q_logs(log_dir):
    paths = sorted(glob.glob(os.path.join(log_dir, "tabular_q_*.csv")))

    if not paths:
        raise FileNotFoundError(f"No Q logs found in {log_dir}")

    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def plot_returns(df, dp_df, out_dir):
    for opponent in sorted(df["villain_policy"].unique()):
        sub = df[df["villain_policy"] == opponent].copy()

        if sub.empty:
            continue

        plt.figure(figsize=(9, 5))

        for obs_mode in sorted(sub["obs_mode"].unique()):
            ss = sub[sub["obs_mode"] == obs_mode].sort_values("episode")
            plt.plot(
                ss["episode"],
                ss["eval_avg_reward"],
                marker="o",
                label=f"Q eval return ({obs_mode})",
            )

        base = dp_df[dp_df["villain_policy"] == opponent]

        if not base.empty:
            dp_ev = float(base["dp_exact_ev"].iloc[0])
            plt.axhline(
                y=dp_ev,
                linestyle="--",
                label=f"DP exact EV = {dp_ev:.3f}",
            )

        plt.title(f"Q-learning return vs DP best response: {opponent}")
        plt.xlabel("Training episode")
        plt.ylabel("Expected/evaluated return")
        plt.legend()
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(out_dir, f"returns_vs_dp_{opponent}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved {out_path}")


def plot_dp_gap(df, dp_df, out_dir):
    merged = df.merge(
        dp_df[["villain_policy", "dp_exact_ev"]],
        on="villain_policy",
        how="left",
    )

    merged["dp_gap"] = merged["dp_exact_ev"] - merged["eval_avg_reward"]

    for opponent in sorted(merged["villain_policy"].unique()):
        sub = merged[merged["villain_policy"] == opponent].copy()

        if sub.empty:
            continue

        plt.figure(figsize=(9, 5))

        for obs_mode in sorted(sub["obs_mode"].unique()):
            ss = sub[sub["obs_mode"] == obs_mode].sort_values("episode")
            plt.plot(
                ss["episode"],
                ss["dp_gap"],
                marker="o",
                label=f"DP gap ({obs_mode})",
            )

        plt.axhline(y=0.0, linestyle="--", label="DP parity")
        plt.title(f"Gap to DP best response: {opponent}")
        plt.xlabel("Training episode")
        plt.ylabel("DP exact EV - Q eval return")
        plt.legend()
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(out_dir, f"gap_to_dp_{opponent}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved {out_path}")


def plot_td_error(df, out_dir):
    for opponent in sorted(df["villain_policy"].unique()):
        sub = df[df["villain_policy"] == opponent].copy()

        if sub.empty:
            continue

        plt.figure(figsize=(9, 5))

        for obs_mode in sorted(sub["obs_mode"].unique()):
            ss = sub[sub["obs_mode"] == obs_mode].sort_values("episode")
            plt.plot(
                ss["episode"],
                ss["avg_abs_td_error_recent"],
                marker="o",
                label=f"avg |TD error| ({obs_mode})",
            )

        plt.title(f"TD error during Q-learning: {opponent}")
        plt.xlabel("Training episode")
        plt.ylabel("Recent average absolute TD error")
        plt.legend()
        plt.grid(True, alpha=0.3)

        out_path = os.path.join(out_dir, f"td_error_{opponent}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-log-dir",
        type=str,
        default="results/leduc/q_learning",
    )
    parser.add_argument(
        "--dp-baselines",
        type=str,
        default="results/leduc/dp_baselines.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/leduc/plots",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_q_logs(args.q_log_dir)
    dp_df = pd.read_csv(args.dp_baselines)

    plot_returns(df, dp_df, args.out_dir)
    plot_dp_gap(df, dp_df, args.out_dir)
    plot_td_error(df, args.out_dir)


if __name__ == "__main__":
    main()