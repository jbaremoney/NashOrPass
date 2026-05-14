import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange
import torch

from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import make_env
from NashOrPass.limit_holdem.agents.modded_dqn import ModdedDQNAgent
from NashOrPass.limit_holdem.agents.factory import make_agent
from NashOrPass.limit_holdem.training.eval import evaluate_agents
from NashOrPass.limit_holdem.utils.seeding import set_seed


def extract_transitions_from_trajectory(trajectory):
    """
    Convert one player's RLCard trajectory into replay-buffer transitions.

    RLCard trajectories are usually:

        state, action, state, action, ..., state

    We extract transitions of the form:

        obs, action, reward, next_obs, done, legal_actions, next_legal_actions

    In RLCard Limit Hold'em, payoff is only known at the end of the hand, so
    intermediate rewards are set to 0.0 and the terminal payoff is assigned
    later in main().
    """
    transitions = []

    for i in range(0, len(trajectory) - 2, 2):
        state = trajectory[i]
        action = trajectory[i + 1]
        next_state = trajectory[i + 2]

        if not isinstance(state, dict) or not isinstance(next_state, dict):
            continue

        done = i + 2 >= len(trajectory) - 1

        transitions.append(
            {
                "obs": state["obs"],
                "action": action,
                "reward": 0.0,
                "next_obs": next_state["obs"],
                "done": done,
                "legal_actions": list(state["legal_actions"].keys()),
                "next_legal_actions": list(next_state["legal_actions"].keys()),
            }
        )

    return transitions


def compute_q_stats(agent, sample_size=512):
    """
    Estimate Q-value magnitudes on a random batch from replay memory.

    Uses min(sample_size, buffer_size), so early evaluations do not produce
    blank CSV fields just because the replay buffer has fewer than 512 samples.
    """
    buffer_size = len(agent.replay_buffer)

    if buffer_size == 0:
        return {
            "q_stat_sample_size": 0,
            "avg_abs_q": np.nan,
            "max_abs_q": np.nan,
            "mean_q": np.nan,
            "min_q": np.nan,
            "max_q": np.nan,
        }

    actual_sample_size = min(sample_size, buffer_size)
    batch = agent.replay_buffer.sample(actual_sample_size)

    obs = torch.tensor(
        batch["obs"],
        dtype=torch.float32,
        device=agent.device,
    )

    agent.q_network.eval()
    with torch.no_grad():
        q = agent.q_network(obs)

        stats = {
            "q_stat_sample_size": actual_sample_size,
            "avg_abs_q": float(q.abs().mean().item()),
            "max_abs_q": float(q.abs().max().item()),
            "mean_q": float(q.mean().item()),
            "min_q": float(q.min().item()),
            "max_q": float(q.max().item()),
        }

    agent.q_network.train()
    return stats


class TrainingModDQNWrapper:
    """
    Wrapper used only during training so we can pass the decayed epsilon into
    the agent while RLCard still sees a normal step/eval_step interface.
    """

    def __init__(self, agent, epsilon=0.0):
        self.agent = agent
        self.epsilon = epsilon
        self.use_raw = False

    def step(self, state):
        obs = state["obs"]
        legal_actions = list(state["legal_actions"].keys())

        return self.agent.select_action(
            obs=obs,
            legal_actions=legal_actions,
            epsilon=self.epsilon,
        )

    def eval_step(self, state):
        action = self.step(state)
        return action, {}


def main():
    parser = argparse.ArgumentParser()

    # Experiment setup.
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--seed", type=int, default=0)

    # Network / optimizer.
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Exploration schedule.
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=20000) # less because policy explores alr

    # Replay/training schedule.
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-steps-per-transition", type=int, default=1)

    # Modded DQN policy options.
    parser.add_argument(
        "--behavior-pol",
        choices=["eps_greedy", "softmax"],
        default="eps_greedy",
        help="Policy used to actually act in the environment.",
    )
    parser.add_argument(
        "--target-pol",
        choices=["greedy", "softmax"],
        default="greedy",
        help="Policy assumed in the Bellman target.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature. Used when behavior-pol or target-pol is softmax.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    run_name = (
        f"mod_dqn_limit_holdem_vs_{args.opponent}"
        f"_seed{args.seed}"
        f"_gamma{args.gamma}"
        f"_lr{args.lr}"
        f"_behavior{args.behavior_pol}"
        f"_target{args.target_pol}"
        f"_temp{args.temperature}"
    )

    save_dir = Path("checkpoints/mod_dqn/limit_holdem") / run_name
    metrics_dir = Path("results/limit_holdem/mod_dqn/metrics") / f"gamma{args.gamma}"

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    env = make_env(seed=args.seed)

    agent = ModdedDQNAgent(
        state_dim=72,
        num_actions=4,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
        behavior_pol=args.behavior_pol,
        target_pol=args.target_pol,
        temperature=args.temperature,
    )

    opponent = make_agent(args.opponent)

    metrics = []
    recent_losses = []
    best_avg_payoff = -float("inf")

    for episode in trange(1, args.episodes + 1):
        frac = min(1.0, episode / args.epsilon_decay_episodes)
        epsilon = args.epsilon_start + frac * (
            args.epsilon_end - args.epsilon_start
        )

        env.set_agents(
            [
                TrainingModDQNWrapper(agent, epsilon=epsilon),
                opponent,
            ]
        )

        trajectories, payoffs = env.run(is_training=True)

        player0_traj = trajectories[0]
        transitions = extract_transitions_from_trajectory(player0_traj)

        # Assign terminal payoff to the last transition.
        if transitions:
            transitions[-1]["reward"] = float(payoffs[0])
            transitions[-1]["done"] = True

        for t in transitions:
            agent.replay_buffer.push(
                t["obs"],
                t["action"],
                t["reward"],
                t["next_obs"],
                t["done"],
                t["legal_actions"],
                t["next_legal_actions"],
            )

            if len(agent.replay_buffer) >= args.learning_starts:
                for _ in range(args.train_steps_per_transition):
                    loss = agent.train_step()
                    if loss is not None:
                        recent_losses.append(loss)

        if episode % args.eval_every == 0:
            eval_env = make_env(seed=args.seed + 123)

            eval_results = evaluate_agents(
                eval_env,
                [agent, opponent],
                num_episodes=args.eval_episodes,
            )

            q_stats = compute_q_stats(agent, sample_size=512)

            row = {
                "episode": episode,
                "epsilon": epsilon,
                "avg_loss": float(np.mean(recent_losses))
                if recent_losses
                else np.nan,
                "num_train_updates": len(recent_losses),
                "avg_payoff": eval_results["avg_payoff"],
                "std_payoff": eval_results["std_payoff"],
                "buffer_size": len(agent.replay_buffer),
                "learning_started": len(agent.replay_buffer) >= args.learning_starts,
                "behavior_pol": args.behavior_pol,
                "target_pol": args.target_pol,
                "temperature": args.temperature,

                # Q-value diagnostics.
                "q_stat_sample_size": q_stats["q_stat_sample_size"],
                "avg_abs_q": q_stats["avg_abs_q"],
                "max_abs_q": q_stats["max_abs_q"],
                "mean_q": q_stats["mean_q"],
                "min_q": q_stats["min_q"],
                "max_q": q_stats["max_q"],
            }

            metrics.append(row)
            print(row)

            if eval_results["avg_payoff"] > best_avg_payoff:
                best_avg_payoff = eval_results["avg_payoff"]
                agent.save(save_dir / "best_model.pt")

            recent_losses = []

            agent.save(save_dir / "model.pt")

            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(save_dir / "metrics.csv", index=False)
            metrics_df.to_csv(metrics_dir / f"{run_name}.csv", index=False)

    agent.save(save_dir / "model.pt")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)
    metrics_df.to_csv(metrics_dir / f"{run_name}.csv", index=False)


if __name__ == "__main__":
    main()