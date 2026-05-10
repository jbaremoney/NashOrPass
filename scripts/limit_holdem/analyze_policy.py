import argparse
from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import make_env
from NashOrPass.limit_holdem.agents.og_dqn import DQNAgent
from NashOrPass.limit_holdem.analysis.saliency import compute_dqn_saliency, summarize_limit_holdem_saliency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    env = make_env(seed=123)
    state, player_id = env.reset()

    agent = DQNAgent()
    agent.load(args.checkpoint)

    saliency = compute_dqn_saliency(agent, state["obs"])
    summary = summarize_limit_holdem_saliency(saliency)

    print("Saliency summary:")
    print(summary)


if __name__ == "__main__":
    main()