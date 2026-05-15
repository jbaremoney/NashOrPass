# scripts/leduc/compare_q_to_dp_occupancy.py

import argparse
import pickle
from collections import Counter

from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_q_and_obs_mode(checkpoint):
    if hasattr(checkpoint, "Q"):
        return checkpoint.Q, getattr(checkpoint, "obs_mode", "perfect")

    if isinstance(checkpoint, dict):
        if "agent" in checkpoint and hasattr(checkpoint["agent"], "Q"):
            agent = checkpoint["agent"]
            return agent.Q, getattr(agent, "obs_mode", checkpoint.get("obs_mode", "perfect"))

        if "Q" in checkpoint:
            return checkpoint["Q"], checkpoint.get("obs_mode", "perfect")

        return checkpoint, "perfect"

    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def perfect_obs_key(state):
    return state.to_tuple()


def imperfect_obs_key(state):
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


def obs_key(state, obs_mode):
    if obs_mode == "perfect":
        return perfect_obs_key(state)
    if obs_mode == "imperfect":
        return imperfect_obs_key(state)
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def q_get(Q, key, action):
    if (key, action) in Q:
        return Q[(key, action)]

    if key in Q and isinstance(Q[key], dict):
        return Q[key].get(action, 0.0)

    return 0.0


def q_greedy_action(Q, state, obs_mode):
    legal = LeducSimpleMDP.legal_actions_from_mdp(state)
    key = obs_key(state, obs_mode)

    # Deterministic tie-break for reproducibility.
    # Important: this tie-break may affect match rates if Q values are tied.
    return max(
        legal,
        key=lambda a: (q_get(Q, key, a), -legal.index(a)),
    )


def traverse(
    state,
    prob,
    Q,
    obs_mode,
    dp_policy,
    villain_policy,
    stats,
):
    if state.to_act == 0:
        state_key = state.to_tuple()

        if state_key not in dp_policy:
            raise KeyError(f"DP policy missing state: {state_key}")

        dp_action = dp_policy[state_key]
        q_action = q_greedy_action(Q, state, obs_mode)

        stats["total_mass"] += prob
        stats["total_states"] += 1

        if q_action == dp_action:
            stats["match_mass"] += prob
            stats["match_states"] += 1
        else:
            stats["mismatch_mass"] += prob
            stats["mismatch_states"] += 1

            if len(stats["examples"]) < 20:
                key = obs_key(state, obs_mode)
                legal = LeducSimpleMDP.legal_actions_from_mdp(state)
                qs = {a: q_get(Q, key, a) for a in legal}
                stats["examples"].append(
                    {
                        "state": state_key,
                        "legal": legal,
                        "dp_action": dp_action,
                        "q_action": q_action,
                        "q_values": qs,
                        "prob": prob,
                    }
                )

        stats["by_street_total"][state.round_stage] += prob
        if q_action == dp_action:
            stats["by_street_match"][state.round_stage] += prob

        stats["by_facing_total"][state.action_facing] += prob
        if q_action == dp_action:
            stats["by_facing_match"][state.action_facing] += prob

        outs = LeducSimpleMDP.action_outcomes(
            state,
            q_action,
            villain_policy=villain_policy,
        )

    else:
        outs = LeducSimpleMDP.env_transition_dist(
            state,
            villain_policy=villain_policy,
        )

    for p, s_next, r, done in outs:
        if not done:
            traverse(
                s_next,
                prob * p,
                Q,
                obs_mode,
                dp_policy,
                villain_policy,
                stats,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-policy-path", required=True)
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--villain-policy", required=True)
    parser.add_argument("--obs-mode", choices=["perfect", "imperfect", "auto"], default="auto")
    args = parser.parse_args()

    dp_policy = load_pickle(args.dp_policy_path)
    q_checkpoint = load_pickle(args.q_path)
    Q, saved_obs_mode = extract_q_and_obs_mode(q_checkpoint)

    obs_mode = saved_obs_mode if args.obs_mode == "auto" else args.obs_mode

    stats = {
        "total_mass": 0.0,
        "match_mass": 0.0,
        "mismatch_mass": 0.0,
        "total_states": 0,
        "match_states": 0,
        "mismatch_states": 0,
        "by_street_total": Counter(),
        "by_street_match": Counter(),
        "by_facing_total": Counter(),
        "by_facing_match": Counter(),
        "examples": [],
    }

    roots = initial_states()
    root_prob = 1.0 / len(roots)

    for root in roots:
        traverse(
            root,
            root_prob,
            Q,
            obs_mode,
            dp_policy,
            args.villain_policy,
            stats,
        )

    mass_match_rate = stats["match_mass"] / stats["total_mass"] if stats["total_mass"] else 0.0
    unweighted_match_rate = stats["match_states"] / stats["total_states"] if stats["total_states"] else 0.0

    print("=" * 80)
    print("Occupancy-Weighted Q Policy vs DP Best Response")
    print("=" * 80)
    print(f"Villain policy:             {args.villain_policy}")
    print(f"Saved obs mode:             {saved_obs_mode}")
    print(f"Used obs mode:              {obs_mode}")
    print(f"Total visited hero states:  {stats['total_states']}")
    print(f"Unweighted match rate:      {unweighted_match_rate:.4f}")
    print(f"Occupancy match rate:       {mass_match_rate:.4f}")
    print(f"Total hero action mass:     {stats['total_mass']:.6f}")
    print(f"Matched action mass:        {stats['match_mass']:.6f}")
    print(f"Mismatched action mass:     {stats['mismatch_mass']:.6f}")

    print("\nBy street, occupancy-weighted:")
    for street in sorted(stats["by_street_total"]):
        total = stats["by_street_total"][street]
        match = stats["by_street_match"][street]
        print(f"  {street:>8}: {match / total:.4f}  mass={total:.6f}")

    print("\nBy facing, occupancy-weighted:")
    for facing in sorted(stats["by_facing_total"]):
        total = stats["by_facing_total"][facing]
        match = stats["by_facing_match"][facing]
        print(f"  {facing:>10}: {match / total:.4f}  mass={total:.6f}")

    print("\nExample mismatches under Q-policy occupancy:")
    for ex in stats["examples"]:
        print("-" * 80)
        print(f"prob:      {ex['prob']:.8f}")
        print(f"state:     {ex['state']}")
        print(f"legal:     {ex['legal']}")
        print(f"DP action: {ex['dp_action']}")
        print(f"Q action:  {ex['q_action']}")
        print(f"Q values:  {ex['q_values']}")


if __name__ == "__main__":
    main()