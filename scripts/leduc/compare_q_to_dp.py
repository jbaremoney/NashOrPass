import argparse
import pickle
from collections import Counter, defaultdict

from NashOrPass.leduc.agents.DP import generate_reachable_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP
from NashOrPass.leduc.agents.tab_q import TabularQAgent


ALL_ACTIONS = [
    "check",
    "bet",
    "call",
    "fold",
    "raise",
    "reraise",
    "utg_call",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_q_agent(path):
    obj = load_pickle(path)

    if isinstance(obj, TabularQAgent):
        return obj

    agent = TabularQAgent(
        actions=ALL_ACTIONS,
        obs_mode=obj.get("obs_mode", "perfect"),
        alpha=obj.get("alpha", 0.1),
        gamma=obj.get("gamma", 1.0),
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=obj.get("seed", 0),
    )
    agent.Q = obj["Q"]
    return agent


def greedy_q_action(agent, state):
    legal_actions = LeducSimpleMDP.legal_actions_from_mdp(state)
    obs_key = agent.obs_key(state)

    best_action = None
    best_q = float("-inf")

    for action in legal_actions:
        q = agent.Q.get((obs_key, action), 0.0)

        if q > best_q:
            best_q = q
            best_action = action

    return best_action, best_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-policy-path", type=str, required=True)
    parser.add_argument("--q-path", type=str, required=True)
    parser.add_argument("--villain-policy", type=str, required=True)
    parser.add_argument("--max-print", type=int, default=30)

    args = parser.parse_args()

    dp_policy = load_pickle(args.dp_policy_path)
    q_agent = load_q_agent(args.q_path)

    states = generate_reachable_states(args.villain_policy)

    hero_states = [
        s for s in states
        if s.to_act == 0 and s.to_tuple() in dp_policy
    ]

    total = 0
    matches = 0
    mismatches = []

    by_street = defaultdict(lambda: Counter({"match": 0, "total": 0}))
    by_facing = defaultdict(lambda: Counter({"match": 0, "total": 0}))

    for state in hero_states:
        dp_action = dp_policy[state.to_tuple()]
        q_action, q_value = greedy_q_action(q_agent, state)

        total += 1
        by_street[state.round_stage]["total"] += 1
        by_facing[state.action_facing]["total"] += 1

        if q_action == dp_action:
            matches += 1
            by_street[state.round_stage]["match"] += 1
            by_facing[state.action_facing]["match"] += 1
        else:
            mismatches.append((state, dp_action, q_action, q_value))

    print("=" * 80)
    print("Q Policy vs DP Best Response")
    print("=" * 80)
    print(f"Villain policy: {args.villain_policy}")
    print(f"Q obs mode:     {q_agent.obs_mode}")
    print(f"Total states:   {total}")
    print(f"Matches:        {matches}")
    print(f"Mismatches:     {total - matches}")
    print(f"Match rate:     {matches / total:.4f}" if total else "Match rate: N/A")

    print("\nBy street:")
    for street, c in sorted(by_street.items()):
        rate = c["match"] / c["total"] if c["total"] else 0.0
        print(f"  {street:>8}: {rate:.4f}  ({c['match']}/{c['total']})")

    print("\nBy facing:")
    for facing, c in sorted(by_facing.items()):
        rate = c["match"] / c["total"] if c["total"] else 0.0
        print(f"  {facing:>12}: {rate:.4f}  ({c['match']}/{c['total']})")

    print("\nExample mismatches:")
    for state, dp_action, q_action, q_value in mismatches[:args.max_print]:
        print("-" * 80)
        print(f"state:     {state.to_tuple()}")
        print(f"legal:     {LeducSimpleMDP.legal_actions_from_mdp(state)}")
        print(f"DP action: {dp_action}")
        print(f"Q action:  {q_action}")
        print(f"Q value:   {q_value}")


if __name__ == "__main__":
    main()