# scripts/leduc/summarize_saved_policies.py

import pickle
from collections import Counter, defaultdict
from pathlib import Path


POLICIES = [
    "uniform",
    "check_call",
    "tight",
    "aggressive",
    "always_raise",
    "always_fold",
    "rank_aware_tight",
    "rank_aware_aggressive",
]


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize_policy(policy):
    total = Counter()
    by_street = defaultdict(Counter)
    by_facing = defaultdict(Counter)

    for state_tuple, action in policy.items():
        total[action] += 1

        round_stage = state_tuple[2]
        action_facing = state_tuple[4]

        by_street[round_stage][action] += 1
        by_facing[action_facing][action] += 1

    return total, by_street, by_facing


def print_counter(counter, indent=""):
    n = sum(counter.values())

    for action, count in sorted(counter.items()):
        pct = 100.0 * count / n if n else 0.0
        print(f"{indent}{action:>8}: {count:5d}  ({pct:6.2f}%)")


def main():
    policy_dir = Path("src/NashOrPass/leduc/agents/saved_policies")

    for name in POLICIES:
        path = policy_dir / f"dp_policy_vs_{name}.pkl"
        policy = load_policy(path)

        total, by_street, by_facing = summarize_policy(policy)

        print("\n" + "=" * 80)
        print(f"Policy: dp_policy_vs_{name}")
        print(f"States: {len(policy)}")
        print("=" * 80)

        print("\nOverall:")
        print_counter(total, indent="  ")

        print("\nBy street:")
        for street, counter in sorted(by_street.items()):
            print(f"  {street}:")
            print_counter(counter, indent="    ")

        print("\nBy action_facing:")
        for facing, counter in sorted(by_facing.items()):
            print(f"  facing={facing}:")
            print_counter(counter, indent="    ")


if __name__ == "__main__":
    main()