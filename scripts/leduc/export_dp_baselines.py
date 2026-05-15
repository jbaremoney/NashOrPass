import argparse
import csv
import os
import pickle

from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


DEFAULT_OPPONENTS = [
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


def evaluate_state_exact(state, policy, villain_policy, gamma=1.0):
    from functools import lru_cache

    tuple_to_state = {state.to_tuple(): state}

    @lru_cache(maxsize=None)
    def V(state_tuple):
        state = tuple_to_state[state_tuple]

        if state.to_act == 0:
            key = state.to_tuple()

            if key not in policy:
                raise KeyError(
                    f"No saved policy action for hero state:\n{key}\n"
                    f"Legal actions: {LeducSimpleMDP.legal_actions_from_mdp(state)}"
                )

            action = policy[key]
            outs = LeducSimpleMDP.action_outcomes(
                state,
                action,
                villain_policy=villain_policy,
            )
        else:
            outs = LeducSimpleMDP.env_transition_dist(
                state,
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


def evaluate_policy_exact(policy, villain_policy, gamma=1.0):
    roots = initial_states()
    values = []

    for root in roots:
        values.append(
            evaluate_state_exact(
                root,
                policy=policy,
                villain_policy=villain_policy,
                gamma=gamma,
            )
        )

    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-dir",
        type=str,
        default="src/NashOrPass/leduc/agents/saved_policies",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="results/leduc/dp_baselines.csv",
    )
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    rows = []

    for opponent in DEFAULT_OPPONENTS:
        policy_path = os.path.join(
            args.policy_dir,
            f"dp_policy_vs_{opponent}.pkl",
        )

        if not os.path.exists(policy_path):
            print(f"Skipping missing policy: {policy_path}")
            continue

        policy = load_policy(policy_path)

        ev = evaluate_policy_exact(
            policy=policy,
            villain_policy=opponent,
            gamma=args.gamma,
        )

        rows.append(
            {
                "villain_policy": opponent,
                "dp_policy_path": policy_path,
                "dp_exact_ev": ev,
            }
        )

        print(f"{opponent:>24}: {ev:.6f}")

    with open(args.out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["villain_policy", "dp_policy_path", "dp_exact_ev"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved DP baselines to {args.out_path}")


if __name__ == "__main__":
    main()