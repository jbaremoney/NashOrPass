"""
change policy from eps-greedy to making prob of action propotional to Q val of it
"""

import random
import torch
import torch.nn.functional as F

from NashOrPass.limit_holdem.networks.mlp_qnet import MLPQNetwork
from NashOrPass.limit_holdem.buffers.replay_buffer import ReplayBuffer

"""
Changed the policy to use est Q values to get probs for each action
can change this to be on or off policy, if off policy, assume strictly greedy future acting
"""
class ModdedDQNAgent:
    def __init__(
        self,
        state_dim=72,
        num_actions=4,
        hidden_dim=256,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=50_000,
        batch_size=64,
        target_update_every=500,
        device=None,
        off_pol = True
    ):
        self.off_pol = off_pol
        self.use_raw = False
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.train_steps = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = MLPQNetwork(state_dim, hidden_dim, num_actions).to(self.device)
        self.target_network = MLPQNetwork(state_dim, hidden_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def _masked_argmax(self, q_values, legal_actions):
        masked = torch.full_like(q_values, -1e9)
        masked[legal_actions] = q_values[legal_actions]
        return int(torch.argmax(masked).item())

    def select_action(self, obs, legal_actions, epsilon=0.0, temperature=1.0):
        """
        Select an action stochastically using Q-values as preferences.

        With probability epsilon, choose a random legal action.
        Otherwise, sample from a softmax distribution over legal-action Q-values.

        Args:
            obs: observation vector
            legal_actions: list of legal action ids
            epsilon: random exploration probability
            temperature: controls how sharp the softmax is
                - high temperature -> more random
                - low temperature -> closer to argmax
        """
        if random.random() < epsilon:
            return random.choice(legal_actions)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(obs_t)[0]

        legal_actions_t = torch.tensor(legal_actions, dtype=torch.long, device=self.device)

        legal_q_values = q_values[legal_actions_t]

        # Convert Q-values into probabilities.
        probs = F.softmax(legal_q_values / temperature, dim=0)

        # Sample one legal action according to those probabilities.
        sampled_index = torch.multinomial(probs, num_samples=1).item()

        return legal_actions[sampled_index]

    def step(self, state, eps=0.0):

        legal_actions = list(state["legal_actions"].keys())
        obs = state["obs"]
        if self.off_pol:
            return self.select_action(obs, legal_actions, epsilon=0.0)
        else:
            return self.select_action(obs, legal_actions, epsilon=eps)

    def eval_step(self, state):
        action = self.step(state)
        return action, {}

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.q_network(obs)
        chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_obs)

            max_next_q = []
            for i, legal in enumerate(batch["next_legal_actions"]):
                if len(legal) == 0:
                    max_next_q.append(torch.tensor(0.0, device=self.device))
                else:
                    masked = torch.full_like(next_q_values[i], -1e9)
                    masked[legal] = next_q_values[i, legal]
                    max_next_q.append(masked.max())

            max_next_q = torch.stack(max_next_q)
            target = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = F.mse_loss(chosen_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.train_steps = checkpoint.get("train_steps", 0)