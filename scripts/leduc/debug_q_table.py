# scripts/leduc/debug_q_table.py

import argparse
import pickle
from collections import Counter

from NashOrPass.leduc.agents.DP import generate_reachable_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_q_and_obs_mode(checkpoint):
    """
    Handles a few possible save formats:

    1. agent object
    2. {"agent": agent, ...}
    3. {"Q": Q, "obs_mode": obs_mode, ...}
    4. raw Q dict
    """
    if hasattr(checkpoint, "Q"):
        return checkpoint.Q, getattr(checkpoint, "obs_mode", "perfect"), checkpoint

    if isinstance(checkpoint, dict):
        if "agent" in checkpoint and hasattr(checkpoint["agent"], "Q"):
            agent = checkpoint["agent"]
            return agent.Q, getattr(agent, "obs_mode", checkpoint.get("obs_mode", "perfect")), agent

        if "Q" in checkpoint:
            return checkpoint["Q"], checkpoint.get("obs_mode", "perfect"), None

        # Maybe the checkpoint itself is the raw Q-table.
        return checkpoint, "unknown", None

    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def perfect_obs_key(state):
    """
    Perfect-information key: same as the full MDP state tuple.
    """
    return state.to_tuple()


def imperfect_obs_key(state):
    """
    Imperfect-information key: hides villain_card.

    Your MDP state tuple looks like:
        (
            hero_card,
            villain_card,
            round_stage,
            flop_card,
            action_facing,
            hero_position,
            hero_raises,
            villain_raises,
            pot,
            terminal,
        )

    So we remove villain_card.
    """
    t = state.to_tuple()

    return (
        t[0],  # hero_card
        t[2],  # round_stage
        t[3],  # flop_card
        t[4],  # action_facing
        t[5],  # hero_position
        t[6],  # hero_raises
        t[7],  # villain_raises
        t[8],  # pot
        t[9],  # terminal
    )


def state_to_obs_key(state, obs_mode, agent=None):
    """
    Prefer the agent's own obs_key/state_to_key method if available.
    Otherwise infer from obs_mode.
    """
    if agent is not None:
        if hasattr(agent, "obs_key"):
            return agent.obs_key(state)
        if hasattr(agent, "state_to_key"):
            return agent.state_to_key(state)

    if obs_mode == "perfect":
        return perfect_obs_key(state)

    if obs_mode == "imperfect":
        return imperfect_obs_key(state)

    # Fallback: most likely your raw Q keys are full state tuples.
    return perfect_obs_key(state)


def q_contains(Q, obs_key, action):
    """
    Supports both common Q-table layouts:

        Q[(obs_key, action)] = value

    and

        Q[obs_key][action] = value
    """
    if (obs_key, action) in Q:
        return True

    if obs_key in Q and isinstance(Q[obs_key], dict) and action in Q[obs_key]:
        return True

    return False


def q_get(Q, obs_key, action):
    if (obs_key, action) in Q:
        return Q[(obs_key, action)]

    if obs_key in Q and isinstance(Q[obs_key], dict):
        return Q[obs_key].get(action, 0.0)

    return 0.0


def print_example_q_keys(Q, n=10):
    print("\nExample raw Q keys:")
    for i, key in enumerate(Q.keys()):
        print(f"  {repr(key)}")
        if i + 1 >= n:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--villain-policy", required=True)
    parser.add_argument(
        "--obs-mode",
        choices=["perfect", "imperfect", "auto"],
        default="auto",
    )
    parser.add_argument("--show-keys", action="store_true")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.q_path)
    Q, saved_obs_mode, agent = extract_q_and_obs_mode(checkpoint)

    obs_mode = saved_obs_mode if args.obs_mode == "auto" else args.obs_mode

    states = generate_reachable_states(args.villain_policy)
    hero_states = [s for s in states if s.to_act == 0]

    total_state_actions = 0
    known_state_actions = 0
    unknown_state_actions = 0
    nonzero_state_actions = 0
    zero_known_state_actions = 0
    fully_unknown_states = 0
    fully_zero_states = 0

    by_street_total = Counter()
    by_street_unknown = Counter()
    by_street_zero_known = Counter()

    examples_unknown = []
    examples_zero = []

    for state in hero_states:
        obs_key = state_to_obs_key(state, obs_mode, agent=agent)
        legal = LeducSimpleMDP.legal_actions_from_mdp(state)

        qs = [q_get(Q, obs_key, a) for a in legal]
        knowns = [q_contains(Q, obs_key, a) for a in legal]

        total_state_actions += len(legal)
        known_state_actions += sum(knowns)
        unknown_state_actions += sum(not x for x in knowns)

        nonzero_state_actions += sum(k and abs(q) > 1e-12 for k, q in zip(knowns, qs))
        zero_known_state_actions += sum(k and abs(q) <= 1e-12 for k, q in zip(knowns, qs))

        by_street_total[state.round_stage] += len(legal)
        by_street_unknown[state.round_stage] += sum(not x for x in knowns)
        by_street_zero_known[state.round_stage] += sum(
            k and abs(q) <= 1e-12 for k, q in zip(knowns, qs)
        )

        if not any(knowns):
            fully_unknown_states += 1
            if len(examples_unknown) < 10:
                examples_unknown.append((state.to_tuple(), obs_key, legal, qs, knowns))

        if all(abs(q) <= 1e-12 for q in qs):
            fully_zero_states += 1
            if len(examples_zero) < 10:
                examples_zero.append((state.to_tuple(), obs_key, legal, qs, knowns))

    print("=" * 80)
    print("Q Table Debug")
    print("=" * 80)
    print(f"Q path:                    {args.q_path}")
    print(f"Villain policy:            {args.villain_policy}")
    print(f"Saved obs mode:            {saved_obs_mode}")
    print(f"Used obs mode:             {obs_mode}")
    print(f"Checkpoint type:           {type(checkpoint)}")
    print(f"Q table type:              {type(Q)}")
    print(f"Raw Q entries:             {len(Q)}")
    print(f"Hero states:               {len(hero_states)}")
    print(f"Legal state-actions:       {total_state_actions}")
    print(f"Known Q entries:           {known_state_actions}")
    print(f"Unknown/default Q entries: {unknown_state_actions}")
    print(f"Nonzero known Q entries:   {nonzero_state_actions}")
    print(f"Zero known Q entries:      {zero_known_state_actions}")
    print(f"Fully unknown states:      {fully_unknown_states}")
    print(f"Fully zero states:         {fully_zero_states}")

    print("\nBy street:")
    for street in sorted(by_street_total):
        total = by_street_total[street]
        unknown = by_street_unknown[street]
        zero_known = by_street_zero_known[street]

        print(
            f"  {street:>8}: "
            f"unknown {unknown}/{total} = {unknown / total:.3f}, "
            f"known-zero {zero_known}/{total} = {zero_known / total:.3f}"
        )

    if args.show_keys:
        print_example_q_keys(Q)

    print("\nExample fully unknown states:")
    for tup, obs_key, legal, qs, knowns in examples_unknown:
        print("-" * 80)
        print(f"state:   {tup}")
        print(f"obs_key: {obs_key}")
        print(f"legal:   {legal}")
        print(f"qs:      {qs}")
        print(f"knowns:  {knowns}")

    print("\nExample fully zero states:")
    for tup, obs_key, legal, qs, knowns in examples_zero:
        print("-" * 80)
        print(f"state:   {tup}")
        print(f"obs_key: {obs_key}")
        print(f"legal:   {legal}")
        print(f"qs:      {qs}")
        print(f"knowns:  {knowns}")


if __name__ == "__main__":
    main()