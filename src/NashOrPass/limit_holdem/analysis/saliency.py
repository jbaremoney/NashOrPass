import torch
import numpy as np


def compute_dqn_saliency(agent, obs, action=None):
    """
    Compute absolute gradient of selected Q-value with respect to observation.
    """
    agent.q_network.eval()

    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    obs_t.requires_grad_(True)

    q_values = agent.q_network(obs_t)

    if action is None:
        action = int(torch.argmax(q_values[0]).item())

    selected_q = q_values[0, action]
    selected_q.backward()

    saliency = obs_t.grad.detach().abs().cpu().numpy()[0]
    return saliency


def summarize_limit_holdem_saliency(saliency):
    """
    RLCard Limit Hold'em uses card features plus betting/action features.
    This gives a coarse grouping, not a perfect semantic explanation.
    """
    saliency = np.asarray(saliency)

    return {
        "card_features_sum": float(saliency[:52].sum()),
        "betting_history_features_sum": float(saliency[52:].sum()),
        "total_sum": float(saliency.sum()),
    }