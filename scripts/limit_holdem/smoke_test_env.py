from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import make_env, get_legal_actions, action_name
from rlcard.agents.random_agent import RandomAgent


def main():
    env = make_env(seed=0)

    print("Environment:", env)
    print("Num players:", env.num_players)
    print("Num actions:", env.num_actions)
    print("State shape:", env.state_shape)

    agents = [
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
    ]
    env.set_agents(agents)

    trajectories, payoffs = env.run(is_training=False)

    print("Payoffs:", payoffs)
    print("Number of player trajectories:", len(trajectories))

    state, player_id = env.reset()
    print("Starting player:", player_id)
    print("Obs shape:", state["obs"].shape)
    print("Legal action ids:", get_legal_actions(state))
    print("Legal actions:", [action_name(a) for a in get_legal_actions(state)])


if __name__ == "__main__":
    main()