import argparse
import csv
import json
import os
import random
from collections import Counter


from NashOrPass.leduc.agents.DP import initial_states
from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP
from NashOrPass.leduc.agents.llm import GPTLeducAgent, ClaudeLeducAgent, GrokLeducAgent


AGGRESSIVE_ACTIONS = {"bet", "raise", "reraise"}


def sample_outcome(outs):
    """
    Sample from a transition distribution of the form:
        [(prob, next_state, reward, done), ...]
    """
    r = random.random()
    total = 0.0

    for p, s_next, reward, done in outs:
        total += p
        if r <= total:
            return s_next, reward, done

    # Floating point fallback.
    p, s_next, reward, done = outs[-1]
    return s_next, reward, done


def make_llm_agent(args):
    if args.agent == "gpt":
        return GPTLeducAgent(
            model_name=args.model_name or "gpt-4.1-mini",
            temperature=args.temperature,
            hist_capacity=args.hist_capacity,
            fallback_policy=args.fallback_policy,
            verbose=args.verbose,
        )

    if args.agent == "claude":
        return ClaudeLeducAgent(
            model_name=args.model_name or "claude-sonnet-4-6",
            temperature=args.temperature,
            hist_capacity=args.hist_capacity,
            fallback_policy=args.fallback_policy,
            verbose=args.verbose,
        )

    if args.agent == "grok":
        return GrokLeducAgent(
            model_name=args.model_name or "grok-4.3",
            temperature=args.temperature,
            hist_capacity=args.hist_capacity,
            fallback_policy=args.fallback_policy,
            verbose=args.verbose,
        )

    raise ValueError(f"Unknown LLM agent: {args.agent}")

def play_leduc_episode(agent, villain_policy="rank_aware_aggressive", max_steps=100):
    """
    Play one sampled Leduc episode.

    The LLM is always treated as hero/player 0.
    The villain is a fixed policy implemented inside LeducSimpleMDP.

    Returns:
        total_reward: payoff for the LLM hero
        trajectory: list of decision records
    """
    state = random.choice(initial_states())
    agent.reset_history()

    total_reward = 0.0
    trajectory = []

    for t in range(max_steps):
        if state.to_act == 0:
            legal_actions = LeducSimpleMDP.legal_actions_from_mdp(state)
            action = agent.choose_action(state)

            info = agent.last_info or {}

            trajectory.append(
                {
                    "t": t,
                    "player": "llm",
                    "state": state.to_tuple(),
                    "legal_actions": legal_actions,
                    "action": action,
                    "bluff": bool(info.get("bluff", False)),
                    "reason": info.get("reason", ""),
                    "pot": getattr(state, "pot", None),
                    "round_stage": getattr(state, "round_stage", None),
                    "action_facing": getattr(state, "action_facing", None),
                    "hero_card": getattr(state, "hero_card", None),
                    "villain_card": getattr(state, "villain_card", None),
                    "flop_card": getattr(state, "flop_card", None),
                }
            )

            outs = LeducSimpleMDP.action_outcomes(
                state,
                action,
                villain_policy=villain_policy,
            )

        else:
            before_state = state

            outs = LeducSimpleMDP.env_transition_dist(
                state,
                villain_policy=villain_policy,
            )

            # We sample first, then infer the villain transition only approximately.
            # For exact villain action logging, we would need MDP.py to return
            # the villain action too.
            trajectory.append(
                {
                    "t": t,
                    "player": "villain_or_chance",
                    "state": before_state.to_tuple(),
                    "villain_policy": villain_policy,
                    "pot": getattr(before_state, "pot", None),
                    "round_stage": getattr(before_state, "round_stage", None),
                    "action_facing": getattr(before_state, "action_facing", None),
                    "hero_card": getattr(before_state, "hero_card", None),
                    "villain_card": getattr(before_state, "villain_card", None),
                    "flop_card": getattr(before_state, "flop_card", None),
                }
            )

        state, reward, done = sample_outcome(outs)
        total_reward += reward

        if done:
            trajectory.append(
                {
                    "t": t + 1,
                    "player": "terminal",
                    "reward": reward,
                    "total_reward": total_reward,
                }
            )
            return total_reward, trajectory

    raise RuntimeError(f"Episode exceeded max_steps={max_steps}. Possible transition bug.")


def load_dp_baseline(villain_policy, path="results/leduc/dp_baselines.csv"):
    """
    Reads the DP baseline CSV created by export_dp_baselines.py.

    Expected format:
        opponent,dp_ev
    or:
        policy,ev

    This is intentionally flexible because your exact column names may differ.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        values = list(row.values())

        # Flexible search: find row containing villain_policy.
        if villain_policy not in values:
            continue

        for key, value in row.items():
            if key is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, default="gpt")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--villain-policy", type=str, default="rank_aware_aggressive")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--hist-capacity", type=int, default=5)
    parser.add_argument("--fallback-policy", type=str, default="check_call")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dp-baselines-path",
        type=str,
        default="results/leduc/dp_baselines.csv",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    if args.out_csv is None:
        args.out_csv = (
            f"results/leduc/llm/"
            f"{args.agent}_vs_{args.villain_policy}_seed{args.seed}.csv"
        )

    if args.out_jsonl is None:
        args.out_jsonl = (
            f"results/leduc/llm/"
            f"{args.agent}_vs_{args.villain_policy}_seed{args.seed}_trajectories.jsonl"
        )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    agent = make_llm_agent(args)

    dp_baseline = load_dp_baseline(
        villain_policy=args.villain_policy,
        path=args.dp_baselines_path,
    )

    episode_rows = []
    trajectory_records = []

    action_counts = Counter()
    total_return = 0.0
    total_llm_decisions = 0
    total_self_reported_bluffs = 0
    total_aggressive_actions = 0

    print("=" * 80)
    print("Leduc LLM Match")
    print("=" * 80)
    print(f"Agent:          {args.agent}")
    print(f"Model:          {agent.model_name}")
    print(f"Villain policy: {args.villain_policy}")
    print(f"Episodes:       {args.episodes}")
    print(f"Temperature:    {args.temperature}")
    print(f"History cap:    {args.hist_capacity}")
    if dp_baseline is not None:
        print(f"DP baseline:    {dp_baseline:.6f}")
    print("-" * 80)

    for ep in range(1, args.episodes + 1):
        payoff, trajectory = play_leduc_episode(
            agent=agent,
            villain_policy=args.villain_policy,
        )

        total_return += payoff

        ep_action_counts = Counter()
        ep_bluffs = 0
        ep_aggressive = 0
        ep_llm_decisions = 0

        for step in trajectory:
            if step.get("player") != "llm":
                continue

            action = step["action"]
            ep_llm_decisions += 1
            ep_action_counts[action] += 1
            action_counts[action] += 1

            if step.get("bluff", False):
                ep_bluffs += 1

            if action in AGGRESSIVE_ACTIONS:
                ep_aggressive += 1

        total_llm_decisions += ep_llm_decisions
        total_self_reported_bluffs += ep_bluffs
        total_aggressive_actions += ep_aggressive

        avg_return = total_return / ep
        gap_to_dp = None if dp_baseline is None else dp_baseline - avg_return

        row = {
            "episode": ep,
            "agent": args.agent,
            "model_name": agent.model_name,
            "villain_policy": args.villain_policy,
            "payoff": payoff,
            "avg_return_so_far": avg_return,
            "dp_baseline": dp_baseline,
            "gap_to_dp": gap_to_dp,
            "llm_decisions": ep_llm_decisions,
            "self_reported_bluffs": ep_bluffs,
            "aggressive_actions": ep_aggressive,
            "fold": ep_action_counts.get("fold", 0),
            "check": ep_action_counts.get("check", 0),
            "call": ep_action_counts.get("call", 0),
            "bet": ep_action_counts.get("bet", 0),
            "raise": ep_action_counts.get("raise", 0),
            "reraise": ep_action_counts.get("reraise", 0),
            "utg_call": ep_action_counts.get("utg_call", 0),
            "invalid_actions_total": agent.stats.get("invalid_actions", 0),
            "api_or_parse_errors_total": agent.stats.get("api_or_parse_errors", 0),
        }

        episode_rows.append(row)

        trajectory_records.append(
            {
                "episode": ep,
                "payoff": payoff,
                "agent": args.agent,
                "model_name": agent.model_name,
                "villain_policy": args.villain_policy,
                "trajectory": trajectory,
            }
        )

        print(
            f"ep={ep:04d} "
            f"payoff={payoff: .3f} "
            f"avg={avg_return: .3f} "
            f"bluffs={ep_bluffs} "
            f"agg={ep_aggressive} "
            f"decisions={ep_llm_decisions}"
        )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    write_jsonl(args.out_jsonl, trajectory_records)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Average payoff:             {total_return / args.episodes:.6f}")

    if dp_baseline is not None:
        print(f"DP best-response baseline:  {dp_baseline:.6f}")
        print(f"Gap to DP:                  {dp_baseline - total_return / args.episodes:.6f}")

    print(f"Total LLM decisions:         {total_llm_decisions}")
    print(f"Action counts:               {dict(action_counts)}")

    bluff_rate = (
        total_self_reported_bluffs / total_llm_decisions
        if total_llm_decisions > 0
        else 0.0
    )
    aggressive_rate = (
        total_aggressive_actions / total_llm_decisions
        if total_llm_decisions > 0
        else 0.0
    )

    print(f"Self-reported bluffs:        {total_self_reported_bluffs}")
    print(f"Self-reported bluff rate:    {bluff_rate:.4f}")
    print(f"Aggressive actions:          {total_aggressive_actions}")
    print(f"Aggressive action rate:      {aggressive_rate:.4f}")
    print(f"Invalid actions:             {agent.stats.get('invalid_actions', 0)}")
    print(f"API/parse errors:            {agent.stats.get('api_or_parse_errors', 0)}")
    print(f"Saved CSV:                   {args.out_csv}")
    print(f"Saved trajectories:          {args.out_jsonl}")


if __name__ == "__main__":
    main()