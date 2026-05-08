import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

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

    for episode in trange(1, args.episodes + 1):
        frac = min(1.0, episode / args.epsilon_decay_episodes)
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        # During training, player 0 is DQN and player 1 is opponent.
        # DQN's eval_step is greedy, so we monkey-patch training step behavior with epsilon.
        class TrainingDQNWrapper:
            def step(self, state):
                obs = state["obs"]
                legal = list(state.get("legal_actions", {}).keys())
                return agent.select_action(obs, legal, epsilon=epsilon)

            def eval_step(self, state):
                return self.step(state), {}

        env.set_agents([TrainingDQNWrapper(), opponent])
        trajectories, payoffs = env.run(is_training=True)

        player0_traj = trajectories[0]
        transitions = extract_transitions_from_trajectory(player0_traj)

        # Assign terminal payoff to the last transition.
        if transitions:
            transitions[-1]["reward"] = float(payoffs[0])
            transitions[-1]["done"] = True

        losses = []
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
                losses.append(loss)

        if episode % args.eval_every == 0:
            eval_env = make_env(seed=args.seed + 123)
            eval_results = evaluate_agents(eval_env, [agent, opponent], num_episodes=500)

            row = {
                "episode": episode,
                "epsilon": epsilon,
                "avg_loss": float(np.mean(losses)) if losses else np.nan,
                "avg_payoff": eval_results["avg_payoff"],
                "std_payoff": eval_results["std_payoff"],
                "buffer_size": len(agent.replay_buffer),
            }
            metrics.append(row)
            print(row)

            agent.save(save_dir / "model.pt")
            pd.DataFrame(metrics).to_csv(save_dir / "metrics.csv", index=False)
            pd.DataFrame(metrics).to_csv(metrics_dir / f"{run_name}.csv", index=False)

    agent.save(save_dir / "model.pt")
    pd.DataFrame(metrics).to_csv(save_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()