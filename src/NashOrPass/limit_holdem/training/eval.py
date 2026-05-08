import numpy as np
from collections import Counter
from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import action_name


def evaluate_agents(env, agents, num_episodes: int = 1000):
    """
    Evaluate a list of agents in an RLCard environment.
    Returns average payoff for player 0 and action counts.
    """
    env.set_agents(agents)

    payoffs = []
    action_counts = Counter()

    for _ in range(num_episodes):
        trajectories, payoff = env.run(is_training=False)
        payoffs.append(payoff[0])

        # RLCard trajectories alternate state/action/state/action...
        # This lightweight logging is defensive and may need adjustment
        # depending on exact RLCard trajectory format.
        for player_traj in trajectories:
            for item in player_traj:
                if isinstance(item, int):
                    action_counts[action_name(item)] += 1

    payoffs = np.array(payoffs, dtype=float)

    return {
        "avg_payoff": float(payoffs.mean()),
        "std_payoff": float(payoffs.std()),
        "num_episodes": num_episodes,
        "action_counts": dict(action_counts),
    }