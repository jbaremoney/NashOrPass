# scripts/leduc/evaluate_policy_occupancy.py

import argparse
import pickle
from collections import Counter, defaultdict

from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def add_count(counter, key, weight):
    counter[key] += weight


def traverse_state(state, policy, villain_policy, prob, counts, by_street, by_facing):
    """
    Traverse the exact game tree under:
        fixed deterministic hero policy
        fixed stochastic/deterministic villain policy

    Adds probability mass to hero action counts whenever hero acts.
    """
    if state.to_act == 0:
        key = state.to_tuple()

        if key not in policy:
            raise KeyError(
                f"Missing policy action for hero state:\n{key}\n"
                f"Legal actions: {LeducSimpleMDP.legal_actions_from_mdp(state)}"
            )

        action = policy[key]

        add_count(counts, action, prob)
        add_count(by_street[state.round_stage], action, prob)
        add_count(by_facing[state.action_facing], action, prob)

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

    for p, s_next, r, done in outs:
        if not done:
            traverse_state(
                s_next,
                policy=policy,
                villain_policy=villain_policy,
                prob=prob * p,
                counts=counts,
                by_street=by_street,
                by_facing=by_facing,
            )


def exact_action_occupancy(policy, villain_policy):
    roots = initial_states()
    root_prob = 1.0 / len(roots)

    counts = Counter()
    by_street = defaultdict(Counter)
    by_facing = defaultdict(Counter)

    for root in roots:
        traverse_state(
            root,
            policy=policy,
            villain_policy=villain_policy,
            prob=root_prob,
            counts=counts,
            by_street=by_street,
            by_facing=by_facing,
        )

    return counts, by_street, by_facing


def print_counter(counter, indent=""):
    total = sum(counter.values())

    for action, mass in sorted(counter.items()):
        pct = 100.0 * mass / total if total else 0.0
        print(f"{indent}{action:>8}: {mass:10.6f}  ({pct:6.2f}%)")

    print(f"{indent}{'total':>8}: {total:10.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--villain-policy", type=str, required=True)

    args = parser.parse_args()

    policy = load_policy(args.policy_path)

    counts, by_street, by_facing = exact_action_occupancy(
        policy=policy,
        villain_policy=args.villain_policy,
    )

    print("=" * 80)
    print("Exact Occupancy-Weighted Action Frequencies")
    print("=" * 80)
    print(f"Policy path:    {args.policy_path}")
    print(f"Villain policy: {args.villain_policy}")

    print("\nOverall:")
    print_counter(counts, indent="  ")

    print("\nBy street:")
    for street, counter in sorted(by_street.items()):
        print(f"  {street}:")
        print_counter(counter, indent="    ")

    print("\nBy facing:")
    for facing, counter in sorted(by_facing.items()):
        print(f"  facing={facing}:")
        print_counter(counter, indent="    ")


if __name__ == "__main__":
    main()