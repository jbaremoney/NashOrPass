from __future__ import annotations

from typing import Any

import numpy as np
import torch


ACTION_ID_TO_NAME = {
    0: "call",
    1: "raise",
    2: "fold",
    3: "check",
}


# RLCard Limit Hold'em card2index.json ordering.
CARD2INDEX = {
    "SA": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4, "S6": 5, "S7": 6,
    "S8": 7, "S9": 8, "ST": 9, "SJ": 10, "SQ": 11, "SK": 12,

    "HA": 13, "H2": 14, "H3": 15, "H4": 16, "H5": 17, "H6": 18,
    "H7": 19, "H8": 20, "H9": 21, "HT": 22, "HJ": 23, "HQ": 24,
    "HK": 25,

    "DA": 26, "D2": 27, "D3": 28, "D4": 29, "D5": 30, "D6": 31,
    "D7": 32, "D8": 33, "D9": 34, "DT": 35, "DJ": 36, "DQ": 37,
    "DK": 38,

    "CA": 39, "C2": 40, "C3": 41, "C4": 42, "C5": 43, "C6": 44,
    "C7": 45, "C8": 46, "C9": 47, "CT": 48, "CJ": 49, "CQ": 50,
    "CK": 51,
}

INDEX2CARD = {v: k for k, v in CARD2INDEX.items()}

ROUND_NAMES = ["preflop", "flop", "turn", "river"]


def get_limit_holdem_feature_names() -> list[str]:
    """
    Return human-readable names for RLCard Limit Hold'em's 72 observation features.

    Layout:
        0:52   card identity features
        52:72  raise-count features

    Important:
        The card features do not by themselves distinguish public cards from
        private cards. They are just card identity indicators for public_cards + hand.
        To separate private/public saliency, pass the full RLCard state/raw_obs
        into summarize_limit_holdem_saliency(...).
    """
    names = []

    for i in range(52):
        card = INDEX2CARD[i]
        names.append(f"card_{card}")

    for round_idx, round_name in enumerate(ROUND_NAMES):
        for raise_count in range(5):
            names.append(f"{round_name}_raise_count_{raise_count}")

    return names


def _extract_raw_obs(state_or_raw_obs: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Accept either:
        1. RLCard extracted state:
            {"obs": ..., "raw_obs": {...}, "legal_actions": ...}

        2. Raw RLCard state:
            {"hand": ..., "public_cards": ..., "raise_nums": ...}

    Return the raw state if available.
    """
    if state_or_raw_obs is None:
        return None

    if "raw_obs" in state_or_raw_obs:
        return state_or_raw_obs["raw_obs"]

    if "hand" in state_or_raw_obs or "public_cards" in state_or_raw_obs:
        return state_or_raw_obs

    return None


def _cards_to_indices(cards: list[Any]) -> list[int]:
    """
    Convert RLCard card strings into feature indices.

    RLCard usually uses strings like:
        "SA", "HT", "D3", "CK"

    This function is defensive in case card objects sneak through.
    """
    indices = []

    for card in cards:
        card_str = str(card)

        if card_str in CARD2INDEX:
            indices.append(CARD2INDEX[card_str])

    return indices


def _safe_sum(x: np.ndarray, indices: list[int]) -> float:
    if len(indices) == 0:
        return 0.0
    return float(x[indices].sum())


def compute_dqn_saliency(
    agent,
    obs,
    action: int | None = None,
    legal_actions: list[int] | None = None,
    absolute: bool = True,
) -> np.ndarray:
    """
    Compute gradient saliency of one selected Q-value with respect to the observation.

    Mathematically, for selected action a:

        saliency_i = |d Q(obs, a) / d obs_i|

    If action is None:
        choose the greedy action.

    If legal_actions is provided:
        choose the greedy legal action.

    Returns:
        A 72-dimensional saliency vector.
    """
    agent.q_network.eval()
    agent.q_network.zero_grad(set_to_none=True)

    obs_t = torch.tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    obs_t.requires_grad_(True)

    q_values = agent.q_network(obs_t)

    if action is None:
        if legal_actions is not None and len(legal_actions) > 0:
            legal_actions_t = torch.tensor(
                legal_actions,
                dtype=torch.long,
                device=agent.device,
            )
            legal_q = q_values[0, legal_actions_t]
            action = int(legal_actions_t[torch.argmax(legal_q)].item())
        else:
            action = int(torch.argmax(q_values[0]).item())

    selected_q = q_values[0, action]
    selected_q.backward()

    grad = obs_t.grad.detach().cpu().numpy()[0]

    if absolute:
        grad = np.abs(grad)

    return grad


def compute_dqn_saliency_details(
    agent,
    obs,
    action: int | None = None,
    legal_actions: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compute saliency plus useful metadata:
        selected action
        selected action name
        q-values
        saliency vector
    """
    agent.q_network.eval()
    agent.q_network.zero_grad(set_to_none=True)

    obs_t = torch.tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    obs_t.requires_grad_(True)

    q_values = agent.q_network(obs_t)
    q_values_np = q_values.detach().cpu().numpy()[0]

    if action is None:
        if legal_actions is not None and len(legal_actions) > 0:
            legal_actions_t = torch.tensor(
                legal_actions,
                dtype=torch.long,
                device=agent.device,
            )
            legal_q = q_values[0, legal_actions_t]
            action = int(legal_actions_t[torch.argmax(legal_q)].item())
        else:
            action = int(torch.argmax(q_values[0]).item())

    selected_q = q_values[0, action]
    selected_q.backward()

    saliency = obs_t.grad.detach().abs().cpu().numpy()[0]

    return {
        "action": int(action),
        "action_name": ACTION_ID_TO_NAME.get(int(action), f"action_{action}"),
        "selected_q": float(selected_q.detach().cpu().item()),
        "q_values": q_values_np,
        "saliency": saliency,
    }


def top_saliency_features(
    saliency,
    top_k: int = 10,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return the top-k most salient observation features.
    """
    saliency = np.asarray(saliency)
    feature_names = feature_names or get_limit_holdem_feature_names()

    top_indices = np.argsort(-saliency)[:top_k]

    return [
        {
            "index": int(i),
            "feature": feature_names[int(i)],
            "saliency": float(saliency[int(i)]),
        }
        for i in top_indices
    ]


def summarize_limit_holdem_saliency(
    saliency,
    state_or_raw_obs: dict[str, Any] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Summarize a 72-dim RLCard Limit Hold'em saliency vector.

    This gives:
        - total card-feature saliency
        - total betting-history saliency
        - active-card saliency
        - inactive-card saliency
        - private/public card saliency, if raw state is available
        - raise-count saliency by street
        - top salient individual features

    Important limitation:
        The 72-dim observation marks public cards and private cards in the same
        52 card identity block. Therefore, from obs alone, you cannot tell whether
        a salient active card was private or public. But if you pass the RLCard
        state/raw_obs, this function can split active card saliency into:
            private_card_features_sum
            public_card_features_sum
    """
    saliency = np.asarray(saliency, dtype=np.float32)

    if saliency.shape[0] != 72:
        raise ValueError(f"Expected saliency with shape (72,), got {saliency.shape}")

    total_sum = float(saliency.sum())
    card_sum = float(saliency[:52].sum())
    betting_sum = float(saliency[52:].sum())

    raw_obs = _extract_raw_obs(state_or_raw_obs)

    private_indices: list[int] = []
    public_indices: list[int] = []

    if raw_obs is not None:
        private_indices = _cards_to_indices(raw_obs.get("hand", []))
        public_indices = _cards_to_indices(raw_obs.get("public_cards", []))

    active_card_indices = sorted(set(private_indices + public_indices))
    inactive_card_indices = [i for i in range(52) if i not in active_card_indices]

    summary: dict[str, Any] = {
        "total_sum": total_sum,

        "card_features_sum": card_sum,
        "betting_history_features_sum": betting_sum,

        "card_features_frac": card_sum / total_sum if total_sum > 0 else np.nan,
        "betting_history_features_frac": betting_sum / total_sum if total_sum > 0 else np.nan,

        "active_card_features_sum": _safe_sum(saliency, active_card_indices),
        "inactive_card_features_sum": _safe_sum(saliency, inactive_card_indices),

        "private_card_features_sum": _safe_sum(saliency, private_indices),
        "public_card_features_sum": _safe_sum(saliency, public_indices),

        "num_private_cards_seen": len(private_indices),
        "num_public_cards_seen": len(public_indices),
    }

    # Add per-round raise-count saliency.
    for round_idx, round_name in enumerate(ROUND_NAMES):
        start = 52 + round_idx * 5
        end = start + 5

        round_sum = float(saliency[start:end].sum())
        summary[f"{round_name}_raise_features_sum"] = round_sum
        summary[f"{round_name}_raise_features_frac"] = (
            round_sum / total_sum if total_sum > 0 else np.nan
        )

    summary["top_features"] = top_saliency_features(
        saliency,
        top_k=top_k,
    )

    return summary


def flatten_saliency_summary(
    summary: dict[str, Any],
    prefix: str = "saliency",
) -> dict[str, float | int]:
    """
    Flatten scalar saliency summary values for CSV logging.

    This intentionally skips "top_features" because that is a list of dicts.
    Save top_features separately as JSON if you want detailed inspection.
    """
    flat = {}

    for key, value in summary.items():
        if key == "top_features":
            continue

        if isinstance(value, (int, float, np.integer, np.floating)):
            flat[f"{prefix}_{key}"] = float(value)

    return flat


def compute_replay_saliency_summary(
    agent,
    sample_size: int = 128,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Compute average saliency summary over observations sampled from replay memory.

    This is useful during training because it answers:

        On states the agent has actually seen, is the Q-network more sensitive
        to card identity features or betting-history features?

    Note:
        ReplayBuffer currently stores obs and legal_actions, but not raw_obs.
        Therefore this function usually cannot split private-card saliency from
        public-card saliency unless you modify the buffer to store raw_obs too.
    """
    buffer_size = len(agent.replay_buffer)

    if buffer_size == 0:
        return {
            "saliency_sample_size": 0,
            "saliency_total_sum": np.nan,
            "saliency_card_features_sum": np.nan,
            "saliency_betting_history_features_sum": np.nan,
            "saliency_card_features_frac": np.nan,
            "saliency_betting_history_features_frac": np.nan,
        }

    actual_sample_size = min(sample_size, buffer_size)
    batch = agent.replay_buffer.sample(actual_sample_size)

    scalar_summaries = []

    for obs, legal_actions in zip(batch["obs"], batch["legal_actions"]):
        details = compute_dqn_saliency_details(
            agent,
            obs,
            action=None,
            legal_actions=legal_actions,
        )

        summary = summarize_limit_holdem_saliency(
            details["saliency"],
            state_or_raw_obs=None,
            top_k=top_k,
        )

        flat = flatten_saliency_summary(summary, prefix="")
        scalar_summaries.append(flat)

    keys = scalar_summaries[0].keys()

    averaged = {
        f"saliency_{key.strip('_')}": float(
            np.nanmean([s[key] for s in scalar_summaries])
        )
        for key in keys
    }

    averaged["saliency_sample_size"] = actual_sample_size

    return averaged