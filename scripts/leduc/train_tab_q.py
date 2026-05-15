import argparse
import csv
import os
import random
from collections import deque

from NashOrPass.leduc.agents.DP import initial_states
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


CSV_FIELDNAMES = [
    "episode",
    "villain_policy",
    "obs_mode",
    "seed",
    "alpha",
    "gamma",
    "epsilon",
    "epsilon_min",
    "epsilon_decay",
    "train_avg_reward_recent",
    "eval_avg_reward",
    "eval_std_reward",
    "avg_abs_td_error_recent",
    "q_states",
    "eval_episodes",
]


def sample_outcome(outcomes, rng):
    """
    Sample from list of:
        (probability, next_state, reward, done)
    """
    x = rng.random()
    total = 0.0

    for p, s_next, r, done in outcomes:
        total += p
        if x <= total:
            return s_next, r, done

    # Numerical fallback.
    p, s_next, r, done = outcomes[-1]
    return s_next, r, done


def advance_until_hero_or_terminal(state, villain_policy, rng):
    """
    If it is villain/chance turn, sample transitions until either:
        - hero acts
        - terminal state is reached

    Returns:
        state, accumulated_reward, done
    """
    total_reward = 0.0
    done = False

    while not done and state.to_act != 0:
        outs = LeducSimpleMDP.env_transition_dist(
            state,
            villain_policy=villain_policy,
        )

        state, r, done = sample_outcome(outs, rng)
        total_reward += r

    return state, total_reward, done


def run_episode(agent, villain_policy, rng, training=True):
    """
    Run one sampled episode.

    Only hero decisions are learned. Villain/chance transitions are sampled
    from the MDP transition model.
    """
    roots = initial_states()
    state = rng.choice(roots)

    state, reward_from_env, done = advance_until_hero_or_terminal(
        state,
        villain_policy=villain_policy,
        rng=rng,
    )

    total_reward = reward_from_env
    td_errors = []

    while not done:
        legal_actions = LeducSimpleMDP.legal_actions_from_mdp(state)

        action = agent.select_action(
            state,
            legal_actions,
            training=training,
        )

        outs = LeducSimpleMDP.action_outcomes(
            state,
            action,
            villain_policy=villain_policy,
        )

        next_state, reward, done = sample_outcome(outs, rng)
        total_reward += reward

        # After hero acts, keep sampling villain/chance until hero acts again
        # or the hand terminates.
        if not done:
            next_state, env_reward, done = advance_until_hero_or_terminal(
                next_state,
                villain_policy=villain_policy,
                rng=rng,
            )
            total_reward += env_reward
            reward += env_reward

        if done:
            next_legal_actions = []
        else:
            next_legal_actions = LeducSimpleMDP.legal_actions_from_mdp(next_state)

        if training:
            td_error = agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_legal_actions=next_legal_actions,
            )
            td_errors.append(abs(td_error))

        state = next_state

    if training:
        agent.decay_epsilon()

    avg_abs_td_error = (
        sum(td_errors) / len(td_errors)
        if td_errors
        else 0.0
    )

    return total_reward, avg_abs_td_error


def evaluate_agent(agent, villain_policy, episodes, seed):
    """
    Monte Carlo evaluation of the current learned Q policy.

    During evaluation, epsilon is temporarily set to zero so the agent acts
    greedily with respect to its current Q table.
    """
    rng = random.Random(seed)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []

    for _ in range(episodes):
        reward, _ = run_episode(
            agent=agent,
            villain_policy=villain_policy,
            rng=rng,
            training=False,
        )
        rewards.append(reward)

    agent.epsilon = old_epsilon

    avg_reward = sum(rewards) / len(rewards)

    variance = sum(
        (r - avg_reward) ** 2
        for r in rewards
    ) / len(rewards)

    return {
        "avg_reward": avg_reward,
        "std_reward": variance ** 0.5,
        "episodes": episodes,
    }


def initialize_csv_log(log_path):
    """
    Create a fresh CSV log file and write the header row.
    """
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_csv_log(log_path, row):
    """
    Append one evaluation row to the CSV log file.
    """
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5_000)

    parser.add_argument("--villain-policy", type=str, default="uniform")
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=["perfect", "imperfect"],
        default="perfect",
    )

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/NashOrPass/leduc/agents/saved_q_agents",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/leduc/q_learning",
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    save_path = os.path.join(
        args.save_dir,
        f"tabular_q_{args.obs_mode}_vs_{args.villain_policy}_seed{args.seed}.pkl",
    )

    log_path = os.path.join(
        args.log_dir,
        f"tabular_q_{args.obs_mode}_vs_{args.villain_policy}_seed{args.seed}.csv",
    )

    initialize_csv_log(log_path)

    agent = TabularQAgent(
        actions=ALL_ACTIONS,
        obs_mode=args.obs_mode,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print("=" * 80)
    print("Training Tabular Q Agent")
    print("=" * 80)
    print(f"Villain policy: {args.villain_policy}")
    print(f"Obs mode:       {args.obs_mode}")
    print(f"Episodes:       {args.episodes}")
    print(f"Eval every:     {args.eval_every}")
    print(f"Eval episodes:  {args.eval_episodes}")
    print(f"Alpha:          {args.alpha}")
    print(f"Gamma:          {args.gamma}")
    print(f"Epsilon:        {args.epsilon}")
    print(f"Epsilon min:    {args.epsilon_min}")
    print(f"Epsilon decay:  {args.epsilon_decay}")
    print(f"Save path:      {save_path}")
    print(f"Log path:       {log_path}")
    print("-" * 80)

    recent_rewards = deque(maxlen=1000)
    recent_td_errors = deque(maxlen=1000)

    for episode in range(1, args.episodes + 1):
        reward, avg_abs_td_error = run_episode(
            agent=agent,
            villain_policy=args.villain_policy,
            rng=rng,
            training=True,
        )

        recent_rewards.append(reward)
        recent_td_errors.append(avg_abs_td_error)

        if episode % args.eval_every == 0:
            train_avg = sum(recent_rewards) / len(recent_rewards)
            td_avg = sum(recent_td_errors) / len(recent_td_errors)

            eval_results = evaluate_agent(
                agent=agent,
                villain_policy=args.villain_policy,
                episodes=args.eval_episodes,
                seed=args.seed + episode,
            )

            row = {
                "episode": episode,
                "villain_policy": args.villain_policy,
                "obs_mode": args.obs_mode,
                "seed": args.seed,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "epsilon": agent.epsilon,
                "epsilon_min": args.epsilon_min,
                "epsilon_decay": args.epsilon_decay,
                "train_avg_reward_recent": train_avg,
                "eval_avg_reward": eval_results["avg_reward"],
                "eval_std_reward": eval_results["std_reward"],
                "avg_abs_td_error_recent": td_avg,
                "q_states": len(agent.Q),
                "eval_episodes": args.eval_episodes,
            }

            append_csv_log(log_path, row)

            print(
                f"episode={episode:>8} "
                f"epsilon={agent.epsilon:.4f} "
                f"train_avg={train_avg: .6f} "
                f"eval_avg={eval_results['avg_reward']: .6f} "
                f"eval_std={eval_results['std_reward']: .6f} "
                f"avg_abs_td={td_avg: .6f} "
                f"q_states={len(agent.Q)}"
            )

    agent.save(save_path)

    print("-" * 80)
    print(f"Saved Q agent to {save_path}")
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()