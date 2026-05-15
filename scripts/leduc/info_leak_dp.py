"""this script shows that the imperfect info leads to different policy than the perfect info one, ie leakage"""

import argparse
import pickle
from collections import defaultdict, Counter

from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def hero_info_key(state):
    """
    Legal hero information state for your current simplified Leduc State.

    Includes only fields that exist on State and that hero is allowed to observe.
    """
    return (
        state.hero_card,
        state.round_stage,
        state.flop_card,
        state.action_facing,
        state.hero_position,
        state.folded_player,
    )


def collect_hero_states(villain_policy):
    """
    Generate all hero decision states reachable under any hero action sequence
    against this fixed villain policy.
    """
    stack = list(initial_states())
    seen = set()
    hero_states = []

    while stack:
        state = stack.pop()
        key = state.to_tuple()

        if key in seen:
            continue

        seen.add(key)

        if state.to_act == 0:
            hero_states.append(state)

            for action in LeducSimpleMDP.legal_actions_from_mdp(state):
                outs = LeducSimpleMDP.action_outcomes(
                    state,
                    action,
                    villain_policy=villain_policy,
                )

                for p, s_next, r, done in outs:
                    if not done:
                        stack.append(s_next)

        else:
            outs = LeducSimpleMDP.env_transition_dist(
                state,
                villain_policy=villain_policy,
            )

            for p, s_next, r, done in outs:
                if not done:
                    stack.append(s_next)

    return hero_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--villain-policy", required=True)
    args = parser.parse_args()

    policy = load_policy(args.policy_path)
    hero_states = collect_hero_states(args.villain_policy)

    info_to_actions = defaultdict(Counter)
    missing = 0

    for state in hero_states:
        state_key = state.to_tuple()

        if state_key not in policy:
            missing += 1
            continue

        info_to_actions[hero_info_key(state)][policy[state_key]] += 1

    conflicts = {
        info_key: action_counts
        for info_key, action_counts in info_to_actions.items()
        if len(action_counts) > 1
    }

    print("=" * 80)
    print("Information-State Policy Leakage Check")
    print("=" * 80)
    print(f"Policy path:       {args.policy_path}")
    print(f"Villain policy:    {args.villain_policy}")
    print(f"Hero states:       {len(hero_states)}")
    print(f"Info states:       {len(info_to_actions)}")
    print(f"Missing states:    {missing}")
    print(f"Conflicting infos: {len(conflicts)}")

    if conflicts:
        print()
        print("Examples where full-state DP chooses different actions")
        print("for the same legal hero information state:")
        print("-" * 80)

        for i, (info_key, action_counts) in enumerate(conflicts.items()):
            if i >= 20:
                break

            print(f"info_key={info_key}")
            print(f"actions={dict(action_counts)}")
            print()

    else:
        print()
        print("No conflicts found. The saved policy is consistent over legal hero information states.")


if __name__ == "__main__":
    main()