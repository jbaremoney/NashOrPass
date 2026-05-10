import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange
import torch

from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import make_env
from NashOrPass.limit_holdem.agents.og_dqn import DQNAgent
from NashOrPass.limit_holdem.agents.factory import make_agent
from NashOrPass.limit_holdem.training.eval import evaluate_agents
from NashOrPass.limit_holdem.utils.seeding import set_seed


def extract_transitions_from_trajectory(trajectory):
    """
    Convert one player's RLCard trajectory into transitions.

    RLCard trajectories are usually:
        state, action, state, action, ..., state

    This function extracts:
        obs, action, reward, next_obs, done, legal_actions, next_legal_actions

    Reward is only known at terminal time, so intermediate rewards are 0.
    """
    transitions = []

    for i in range(0, len(trajectory) - 2, 2):
        state = trajectory[i]
        action = trajectory[i + 1]
        next_state = trajectory[i + 2]

        if not isinstance(state, dict) or not isinstance(next_state, dict):
            continue

        done = i + 2 >= len(trajectory) - 1

        transitions.append({
            "obs": state["obs"],
            "action": action,
            "reward": 0.0,
            "next_obs": next_state["obs"],
            "done": done,
            "legal_actions": list(state["legal_actions"].keys()),
            "next_legal_actions": list(next_state["legal_actions"].keys()),
        })

    return transitions


def compute_q_stats(agent, sample_size=512):
    """
    Estimate Q-value magnitudes on a random batch from the replay buffer.

    Returns:
        avg_abs_q: average absolute Q-value
        max_abs_q: maximum absolute Q-value
        mean_q: average raw Q-value
        min_q: minimum Q-value
        max_q: maximum Q-value
    """
    if len(agent.replay_buffer) < sample_size:
        return {
            "avg_abs_q": np.nan,
            "max_abs_q": np.nan,
            "mean_q": np.nan,
            "min_q": np.nan,
            "max_q": np.nan,
        }

    batch = agent.replay_buffer.sample(sample_size)

    obs = torch.tensor(
        batch["obs"],
        dtype=torch.float32,
        device=agent.device,
    )

    agent.q_network.eval()
    with torch.no_grad():
        q = agent.q_network(obs)

        stats = {
            "avg_abs_q": float(q.abs().mean().item()),
            "max_abs_q": float(q.abs().max().item()),
            "mean_q": float(q.mean().item()),
            "min_q": float(q.min().item()),
            "max_q": float(q.max().item()),
        }

    agent.q_network.train()
    return stats


class TrainingDQNWrapper:
    def __init__(self, agent, epsilon=0.0):
        self.agent = agent
        self.epsilon = epsilon
        self.use_raw = False

    def step(self, state):
        obs = state["obs"]
        legal = list(state["legal_actions"].keys())
        return self.agent.select_action(obs, legal, epsilon=self.epsilon)

    def eval_step(self, state):
        return self.step(state), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=3000)
    args = parser.parse_args()

    set_seed(args.seed)

    run_name = f"dqn_limit_holdem_vs_{args.opponent}_seed{args.seed}"
    save_dir = Path("checkpoints/dqn/limit_holdem") / run_name
    metrics_dir = Path("results/metrics")
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    env = make_env(seed=args.seed)
    agent = DQNAgent(
        state_dim=72,
        num_actions=4,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lr=args.lr,
    )
    opponent = make_agent(args.opponent)

    metrics = []
    recent_losses = []  # losses since the last evaluation

    for episode in trange(1, args.episodes + 1):
        frac = min(1.0, episode / args.epsilon_decay_episodes)
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        env.set_agents([TrainingDQNWrapper(agent, epsilon=epsilon), opponent])
        trajectories, payoffs = env.run(is_training=True)

        player0_traj = trajectories[0]
        transitions = extract_transitions_from_trajectory(player0_traj)

        # Assign terminal payoff to the last transition.
        if transitions:
            transitions[-1]["reward"] = float(payoffs[0])
            transitions[-1]["done"] = True

        # transitions within 1 episode
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

            loss = agent.train_step()
            if loss is not None:
                recent_losses.append(loss)

        if episode % args.eval_every == 0:
            eval_env = make_env(seed=args.seed + 123)
            eval_results = evaluate_agents(eval_env, [agent, opponent], num_episodes=500)

            q_stats = compute_q_stats(agent, sample_size=512)

            row = {
                "episode": episode,
                "epsilon": epsilon,
                "avg_loss": float(np.mean(recent_losses)) if recent_losses else np.nan,
                "num_train_updates": len(recent_losses),
                "avg_payoff": eval_results["avg_payoff"],
                "std_payoff": eval_results["std_payoff"],
                "buffer_size": len(agent.replay_buffer),

                # Q-value diagnostics
                "avg_abs_q": q_stats["avg_abs_q"],
                "max_abs_q": q_stats["max_abs_q"],
                "mean_q": q_stats["mean_q"],
                "min_q": q_stats["min_q"],
                "max_q": q_stats["max_q"],
            }

            metrics.append(row)
            print(row)

            recent_losses = []

            agent.save(save_dir / "model.pt")
            pd.DataFrame(metrics).to_csv(save_dir / "metrics.csv", index=False)
            pd.DataFrame(metrics).to_csv(metrics_dir / f"{run_name}.csv", index=False)

    agent.save(save_dir / "model.pt")
    pd.DataFrame(metrics).to_csv(save_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()