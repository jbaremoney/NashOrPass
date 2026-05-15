import argparse
import glob
import os
import re

import pandas as pd


def parse_filename(path):
    """
    Expected filenames like:
        results/leduc/llm/gpt_vs_uniform_seed0.csv
        results/leduc/llm/claude_vs_rank_aware_aggressive_seed0.csv
    """
    name = os.path.basename(path)
    match = re.match(r"(.+)_vs_(.+)_seed(\d+)\.csv", name)

    if not match:
        return None

    agent = match.group(1)
    opponent = match.group(2)
    seed = int(match.group(3))

    return agent, opponent, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-dir",
        type=str,
        default="results/leduc/llm",
    )
    parser.add_argument(
        "--dp-baselines",
        type=str,
        default="results/leduc/dp_baselines.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/leduc/llm_summary.csv",
    )

    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.llm_dir, "*.csv")))

    if not paths:
        raise FileNotFoundError(f"No LLM CSV files found in {args.llm_dir}")

    dp_df = pd.read_csv(args.dp_baselines)

    # Supports either:
    #   villain_policy,dp_ev
    # or:
    #   policy,ev
    if "villain_policy" in dp_df.columns:
        dp_key_col = "villain_policy"
    elif "policy" in dp_df.columns:
        dp_key_col = "policy"
    else:
        dp_key_col = dp_df.columns[0]

    if "dp_ev" in dp_df.columns:
        dp_val_col = "dp_ev"
    elif "ev" in dp_df.columns:
        dp_val_col = "ev"
    elif "avg_ev" in dp_df.columns:
        dp_val_col = "avg_ev"
    else:
        dp_val_col = dp_df.columns[1]

    dp_map = dict(zip(dp_df[dp_key_col], dp_df[dp_val_col]))

    rows = []

    for path in paths:
        parsed = parse_filename(path)
        if parsed is None:
            print(f"Skipping unrecognized filename: {path}")
            continue

        agent, opponent, seed = parsed
        df = pd.read_csv(path)

        avg_payoff = df["payoff"].mean()
        std_payoff = df["payoff"].std()
        episodes = len(df)

        total_decisions = df["decisions"].sum() if "decisions" in df.columns else None
        total_bluffs = df["bluffs"].sum() if "bluffs" in df.columns else None
        total_agg = df["aggressive_actions"].sum() if "aggressive_actions" in df.columns else None

        bluff_rate = (
            total_bluffs / total_decisions
            if total_decisions and total_decisions > 0
            else 0.0
        )

        aggressive_rate = (
            total_agg / total_decisions
            if total_decisions and total_decisions > 0
            else 0.0
        )

        dp_ev = dp_map.get(opponent, None)
        gap_to_dp = None if dp_ev is None else dp_ev - avg_payoff

        rows.append({
            "agent": agent,
            "opponent": opponent,
            "seed": seed,
            "episodes": episodes,
            "avg_payoff": avg_payoff,
            "std_payoff": std_payoff,
            "dp_best_response_ev": dp_ev,
            "gap_to_dp": gap_to_dp,
            "total_decisions": total_decisions,
            "total_bluffs": total_bluffs,
            "bluff_rate": bluff_rate,
            "aggressive_rate": aggressive_rate,
            "csv": path,
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["opponent", "avg_payoff"], ascending=[True, False])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.out, index=False)

    print("=" * 100)
    print("LLM Summary")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()