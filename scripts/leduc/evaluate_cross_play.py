import argparse
import csv
import pickle
from functools import lru_cache
from pathlib import Path

from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


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


def evaluate_state_exact(state, policy, villain_policy="uniform", gamma=1.0):
    """
    Exact EV of a saved deterministic hero policy from one full MDP state.
    """

    tuple_to_state = {state.to_tuple(): state}

    @lru_cache(maxsize=None)
    def V(state_tuple):
        state_obj = tuple_to_state[state_tuple]

        if state_obj.to_act == 0:
            key = state_obj.to_tuple()

            if key not in policy:
                raise KeyError(
                    f"No saved policy action for hero state:\n{key}\n"
                    f"Legal actions: {LeducSimpleMDP.legal_actions_from_mdp(state_obj)}"
                )

            action = policy[key]
            outs = LeducSimpleMDP.action_outcomes(
                state_obj,
                action,
                villain_policy=villain_policy,
            )
        else:
            outs = LeducSimpleMDP.env_transition_dist(
                state_obj,
                villain_policy=villain_policy,
            )

        total = 0.0

        for p, s_next, r, done in outs:
            if done:
                total += p * r
            else:
                tuple_to_state[s_next.to_tuple()] = s_next
                total += p * (r + gamma * V(s_next.to_tuple()))

        return total

    return V(state.to_tuple())


def evaluate_policy_exact(policy, villain_policy="uniform", gamma=1.0):
    roots = initial_states()

    values = []
    by_position = {"btn": [], "bb": []}

    for root in roots:
        ev = evaluate_state_exact(
            root,
            policy=policy,
            villain_policy=villain_policy,
            gamma=gamma,
        )

        values.append(ev)
        by_position[root.hero_position].append(ev)

    return {
        "num_roots": len(roots),
        "avg_ev": sum(values) / len(values),
        "btn_ev": sum(by_position["btn"]) / len(by_position["btn"]),
        "bb_ev": sum(by_position["bb"]) / len(by_position["bb"]),
        "min_root_ev": min(values),
        "max_root_ev": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-dir",
        type=str,
        default="src/NashOrPass/leduc/agents/saved_policies",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/leduc/dp_cross_play.csv",
    )

    args = parser.parse_args()

    policy_dir = Path(args.policy_dir)
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        ["trained_vs"]
        + [f"eval_{p}" for p in POLICIES]
        + ["diagonal_ev", "mean_ev", "worst_ev", "best_ev"]
    )

    rows = []

    print("=" * 100)
    print("DP Cross-Play Matrix")
    print("=" * 100)
    print(",".join(header))

    for trained_vs in POLICIES:
        policy_path = policy_dir / f"dp_policy_vs_{trained_vs}.pkl"

        if not policy_path.exists():
            raise FileNotFoundError(
                f"Missing policy file: {policy_path}\n"
                f"Run scripts/leduc/run_dp_all_oppts.sh first."
            )

        policy = load_policy(policy_path)

        row = [trained_vs]
        numeric_values = []

        for eval_vs in POLICIES:
            results = evaluate_policy_exact(
                policy=policy,
                villain_policy=eval_vs,
                gamma=args.gamma,
            )

            ev = results["avg_ev"]
            numeric_values.append(ev)
            row.append(f"{ev:.6f}")

        diagonal_index = POLICIES.index(trained_vs)
        diagonal_ev = numeric_values[diagonal_index]
        mean_ev = sum(numeric_values) / len(numeric_values)
        worst_ev = min(numeric_values)
        best_ev = max(numeric_values)

        row += [
            f"{diagonal_ev:.6f}",
            f"{mean_ev:.6f}",
            f"{worst_ev:.6f}",
            f"{best_ev:.6f}",
        ]

        rows.append(row)
        print(",".join(row))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print()
    print(f"Saved cross-play CSV to: {csv_path}")


if __name__ == "__main__":
    main()