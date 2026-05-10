from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


DEFAULT_ACTION_ID_TO_NAME = {
    0: "call",
    1: "raise",
    2: "fold",
    3: "check",
}


def action_counts_to_freqs(
    action_counts: dict[str, int],
) -> dict[str, float]:
    """
    Convert action counts into action frequencies.

    Example:
        {"call": 10, "raise": 5} -> {"call_freq": 0.666..., "raise_freq": 0.333...}
    """
    total = sum(action_counts.values())

    if total == 0:
        return {f"{action}_freq": np.nan for action in action_counts}

    return {
        f"{action}_freq": count / total
        for action, count in action_counts.items()
    }


def compute_q_action_stats(
    agent,
    obs_batch: np.ndarray,
    legal_actions_batch: list[list[int]] | None = None,
    action_id_to_name: dict[int, str] | None = None,
) -> dict[str, float]:
    """
    Compute basic Q-value and greedy-policy diagnostics over a batch of observations.

    If legal_actions_batch is provided, greedy actions are computed only over legal
    actions. If not provided, argmax is taken over all network outputs.

    Returns stats like:
        mean_q_call
        mean_q_raise
        greedy_call_count
        greedy_call_freq
        avg_policy_margin

    The policy margin is:
        best legal Q - second-best legal Q

    Large margin means the network strongly prefers its chosen action.
    Small margin means the action choice is fragile/uncertain.
    """
    action_id_to_name = action_id_to_name or DEFAULT_ACTION_ID_TO_NAME

    if len(obs_batch) == 0:
        out = {}
        for action_id, name in action_id_to_name.items():
            out[f"mean_q_{name}"] = np.nan
            out[f"greedy_{name}_count"] = 0
            out[f"greedy_{name}_freq"] = np.nan
        out["avg_policy_margin"] = np.nan
        return out

    obs_t = torch.tensor(
        obs_batch,
        dtype=torch.float32,
        device=agent.device,
    )

    agent.q_network.eval()
    with torch.no_grad():
        q_values = agent.q_network(obs_t).detach().cpu().numpy()

    # Mean raw Q-value for each action output.
    stats: dict[str, float] = {}
    for action_id, name in action_id_to_name.items():
        stats[f"mean_q_{name}"] = float(np.mean(q_values[:, action_id]))

    greedy_counts = Counter()
    margins = []

    for i in range(len(q_values)):
        q = q_values[i]

        if legal_actions_batch is None:
            legal_actions = list(action_id_to_name.keys())
        else:
            legal_actions = list(legal_actions_batch[i])

        if len(legal_actions) == 0:
            continue

        legal_q = np.array([q[a] for a in legal_actions], dtype=np.float32)
        best_local_idx = int(np.argmax(legal_q))
        best_action = int(legal_actions[best_local_idx])

        greedy_counts[best_action] += 1

        if len(legal_q) >= 2:
            sorted_q = np.sort(legal_q)
            margin = float(sorted_q[-1] - sorted_q[-2])
            margins.append(margin)

    total = sum(greedy_counts.values())

    for action_id, name in action_id_to_name.items():
        count = int(greedy_counts[action_id])
        stats[f"greedy_{name}_count"] = count
        stats[f"greedy_{name}_freq"] = count / total if total > 0 else np.nan

    stats["avg_policy_margin"] = float(np.mean(margins)) if margins else np.nan

    return stats


def compute_replay_policy_frequencies(
    agent,
    sample_size: int = 512,
    action_id_to_name: dict[int, str] | None = None,
) -> dict[str, float]:
    """
    Analyze the current greedy policy on observations sampled from replay memory.

    This does not play new hands. It asks:

        On states the agent has seen before, what action would the current
        Q-network greedily choose?

    This is useful during training because it tells you whether the learned
    policy is becoming call-heavy, raise-heavy, fold-heavy, etc.
    """
    action_id_to_name = action_id_to_name or DEFAULT_ACTION_ID_TO_NAME

    buffer_size = len(agent.replay_buffer)

    if buffer_size == 0:
        stats = {
            "policy_sample_size": 0,
            "avg_policy_margin": np.nan,
        }
        for action_id, name in action_id_to_name.items():
            stats[f"mean_q_{name}"] = np.nan
            stats[f"greedy_{name}_count"] = 0
            stats[f"greedy_{name}_freq"] = np.nan
        return stats

    actual_sample_size = min(sample_size, buffer_size)
    batch = agent.replay_buffer.sample(actual_sample_size)

    stats = compute_q_action_stats(
        agent=agent,
        obs_batch=batch["obs"],
        legal_actions_batch=batch["legal_actions"],
        action_id_to_name=action_id_to_name,
    )

    stats["policy_sample_size"] = actual_sample_size
    return stats


def evaluate_action_frequencies(
    env,
    agents: list[Any],
    num_episodes: int = 500,
    player_id: int = 0,
    action_id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Run evaluation episodes and count the actions taken by one player.

    This is direct policy analysis: it tells you what the agent actually does
    while playing, not merely what it would do on replay-buffer states.

    Returns:
        avg_payoff
        std_payoff
        action_counts
        action frequencies
        total_actions
        num_episodes

    Assumes RLCard/PettingZoo-style env.run returns:
        trajectories, payoffs

    and that trajectories[player_id] alternates like:
        state, action, state, action, ..., state
    """
    action_id_to_name = action_id_to_name or DEFAULT_ACTION_ID_TO_NAME

    env.set_agents(agents)

    payoffs = []
    action_counts = Counter()

    for _ in range(num_episodes):
        trajectories, payoff = env.run(is_training=False)

        payoffs.append(float(payoff[player_id]))

        traj = trajectories[player_id]

        # RLCard trajectories are usually:
        # state, action, state, action, ..., state
        for i in range(1, len(traj), 2):
            action = traj[i]

            if isinstance(action, (int, np.integer)):
                action_id = int(action)
                action_name = action_id_to_name.get(action_id, f"action_{action_id}")
                action_counts[action_name] += 1

    # Ensure all known actions appear, even if count is zero.
    named_counts = {
        name: int(action_counts.get(name, 0))
        for _, name in sorted(action_id_to_name.items())
    }

    freqs = action_counts_to_freqs(named_counts)

    result: dict[str, Any] = {
        "avg_payoff": float(np.mean(payoffs)) if payoffs else np.nan,
        "std_payoff": float(np.std(payoffs)) if payoffs else np.nan,
        "num_episodes": int(num_episodes),
        "total_actions": int(sum(named_counts.values())),
        "action_counts": named_counts,
    }

    result.update(freqs)
    return result


def flatten_action_frequency_results(
    results: dict[str, Any],
    prefix: str = "eval",
) -> dict[str, float | int]:
    """
    Flatten nested action-frequency results so they can be inserted into a CSV row.

    Example:
        {
            "action_counts": {"call": 10, "raise": 5},
            "call_freq": 0.66,
        }

    becomes:
        {
            "eval_call_count": 10,
            "eval_raise_count": 5,
            "eval_call_freq": 0.66,
        }
    """
    flat: dict[str, float | int] = {}

    if "action_counts" in results:
        for action_name, count in results["action_counts"].items():
            flat[f"{prefix}_{action_name}_count"] = int(count)

    for key, value in results.items():
        if key == "action_counts":
            continue

        if isinstance(value, (int, float, np.integer, np.floating)):
            flat[f"{prefix}_{key}"] = float(value)

    return flat