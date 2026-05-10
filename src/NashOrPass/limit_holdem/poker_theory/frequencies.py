from collections import Counter
from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import action_name


def count_actions_from_trajectories(trajectories):
    counts = Counter()

    for traj in trajectories:
        for item in traj:
            if isinstance(item, int):
                counts[action_name(item)] += 1

    return dict(counts)


def normalize_counts(counts):
    total = sum(counts.values())

    if total == 0:
        return {k: 0.0 for k in counts}

    return {k: v / total for k, v in counts.items()}