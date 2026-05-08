import argparse
from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import make_env
from NashOrPass.limit_holdem.agents.factory import make_agent
from NashOrPass.limit_holdem.training.eval import evaluate_agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="random")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    env = make_env(seed=0)

    agent = make_agent(args.agent)
    opponent = make_agent(args.opponent)

    results = evaluate_agents(env, [agent, opponent], num_episodes=args.episodes)

    print(results)


if __name__ == "__main__":
    main()