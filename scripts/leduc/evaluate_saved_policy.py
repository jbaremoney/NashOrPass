

import argparse
import pickle
from functools import lru_cache

from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


def load_policy(path):
    """
    Loads saved DP policy.

    Your save_policy function saved:
        {state.to_tuple(): action}
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_state_exact(state, policy, villain_policy="standard", gamma=1.0):
    """
    Exact expected value of a saved deterministic hero policy from a state.

    The policy is expected to map:
        state.to_tuple() -> action

    Returns expected payoff for hero.
    """

    @lru_cache(maxsize=None)
    def V(state_tuple):
        # Reconstructing from tuple would require a from_tuple method.
        # So instead we close over an auxiliary object-cache below.
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

    tuple_to_state = {state.to_tuple(): state}
    return V(state.to_tuple())


def evaluate_policy_exact(policy, villain_policy="standard", gamma=1.0):
    """
    Exact EV from the initial deal distribution.

    Your initial_states() enumerates all private card assignments and both
    hero positions. It creates 6 * 5 * 2 = 60 equally likely starting states.
    """
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

    avg_ev = sum(values) / len(values)

    return {
        "num_roots": len(roots),
        "avg_ev": avg_ev,
        "btn_ev": sum(by_position["btn"]) / len(by_position["btn"]),
        "bb_ev": sum(by_position["bb"]) / len(by_position["bb"]),
        "min_root_ev": min(values),
        "max_root_ev": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-path",
        type=str,
        default="src/NashOrPass/leduc/agents/saved_policies/dp_policy_vs_standard.pkl",
    )
    parser.add_argument("--villain-policy", type=str, default="standard")
    parser.add_argument("--gamma", type=float, default=1.0)

    args = parser.parse_args()

    policy = load_policy(args.policy_path)

    results = evaluate_policy_exact(
        policy=policy,
        villain_policy=args.villain_policy,
        gamma=args.gamma,
    )

    print("=" * 70)
    print("Saved DP Policy Evaluation")
    print("=" * 70)
    print(f"Policy path:     {args.policy_path}")
    print(f"Villain policy:  {args.villain_policy}")
    print(f"Gamma:           {args.gamma}")
    print("-" * 70)

    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()