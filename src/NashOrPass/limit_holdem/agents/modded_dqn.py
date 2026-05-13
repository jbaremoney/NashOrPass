import random

import torch
import torch.nn.functional as F

from NashOrPass.limit_holdem.networks.mlp_qnet import MLPQNetwork
from NashOrPass.limit_holdem.buffers.replay_buffer import ReplayBuffer


class ModdedDQNAgent:
    """
    DQN with separately configurable behavior and target policies

    Behavior policy:
        The policy used to actually act in the environment and generate data.

        "eps_greedy":
            With probability epsilon, choose random legal action.
            Otherwise, choose argmax_a Q(s, a).

        "softmax":
            With probability epsilon, choose random legal action.
            Otherwise, sample from softmax(Q(s, a) / temperature) over legal actions.

    Target policy:
        The policy assumed in the Bellman target.

        "greedy":
            Standard DQN / Q-learning target:
                y = r + gamma * max_a' Q_target(s', a')

        "softmax":
            Expected-SARSA-style soft target:
                y = r + gamma * E_{a'~softmax(Q_target(s')/temperature)}
                                  [Q_target(s', a')]

    Good default configurations:

        Standard DQN:
            behavior_pol="eps_greedy"
            target_pol="greedy"

        Soft Expected SARSA-ish:
            behavior_pol="softmax"
            target_pol="softmax"
    """

    VALID_BEHAVIOR_POLICIES = {"eps_greedy", "softmax"}
    VALID_TARGET_POLICIES = {"greedy", "softmax", "eps_greedy"}

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
        behavior_pol: str = "eps_greedy",
        target_pol: str = "greedy",
        temperature: float = 1.0,
    ):
        if behavior_pol not in self.VALID_BEHAVIOR_POLICIES:
            raise ValueError(
                f"Unknown behavior_pol={behavior_pol}. "
                f"Expected one of {self.VALID_BEHAVIOR_POLICIES}."
            )

        if target_pol not in self.VALID_TARGET_POLICIES:
            raise ValueError(
                f"Unknown target_pol={target_pol}. "
                f"Expected one of {self.VALID_TARGET_POLICIES}."
            )

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.behavior_pol = behavior_pol
        self.target_pol = target_pol
        self.temperature = float(temperature)

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
        """
        Argmax over legal actions only.
        """
        masked = torch.full_like(q_values, -1e9)
        masked[legal_actions] = q_values[legal_actions]
        return int(torch.argmax(masked).item())

    def _legal_softmax_probs(self, q_values, legal_actions, temperature=None):
        """
        Convert legal-action Q-values into a probability distribution.

        q_values:
            Tensor of shape (num_actions,)

        legal_actions:
            list[int]

        Returns:
            legal_actions_t:
                Tensor containing legal action ids.

            probs:
                Tensor of probabilities over legal_actions_t.
        """
        if temperature is None:
            temperature = self.temperature

        legal_actions_t = torch.tensor(
            legal_actions,
            dtype=torch.long,
            device=self.device,
        )

        legal_q_values = q_values[legal_actions_t]
        probs = F.softmax(legal_q_values / temperature, dim=0)

        return legal_actions_t, probs

    def select_eps_greedy_action(self, obs, legal_actions, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        """
        if random.random() < epsilon:
            return random.choice(legal_actions)

        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(obs_t)[0]

        return self._masked_argmax(q_values, legal_actions)

    def select_softmax_action(self, obs, legal_actions, epsilon=0.0, temperature=None):
        """
        Select action using epsilon-softmax over legal Q-values.

        With probability epsilon:
            choose random legal action.

        Otherwise:
            sample from softmax(Q(s, a) / temperature) over legal actions.
        """
        if temperature is None:
            temperature = self.temperature

        if random.random() < epsilon:
            return random.choice(legal_actions)

        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(obs_t)[0]

        # Very low temperature approximates greedy action selection.
        if temperature <= 1e-8:
            return self._masked_argmax(q_values, legal_actions)

        legal_actions_t, probs = self._legal_softmax_probs(
            q_values=q_values,
            legal_actions=legal_actions,
            temperature=temperature,
        )

        sampled_index = torch.multinomial(probs, num_samples=1).item()
        action = int(legal_actions_t[sampled_index].item())

        return action

    def select_action(self, obs, legal_actions, epsilon=0.0):
        """
        Select action according to the behavior policy.

        This is the actual policy used to play hands.
        """
        if self.behavior_pol == "eps_greedy":
            return self.select_eps_greedy_action(
                obs=obs,
                legal_actions=legal_actions,
                epsilon=epsilon,
            )

        if self.behavior_pol == "softmax":
            return self.select_softmax_action(
                obs=obs,
                legal_actions=legal_actions,
                epsilon=epsilon,
                temperature=self.temperature,
            )

        raise ValueError(f"Unknown behavior_pol: {self.behavior_pol}")

    def step(self, state, eps=0.0):
        """
        RLCard calls this to get the agent's actual environment action.

        This uses behavior_pol, not target_pol.
        """
        legal_actions = list(state["legal_actions"].keys())
        obs = state["obs"]

        return self.select_action(
            obs=obs,
            legal_actions=legal_actions,
            epsilon=eps,
        )

    def eval_step(self, state):
        """
        Evaluation-time action.

        For now this uses the same behavior policy with epsilon=0.

        So:
            behavior_pol="eps_greedy" -> greedy evaluation
            behavior_pol="softmax"    -> softmax sampling evaluation

        """
        legal_actions = list(state["legal_actions"].keys())
        obs = state["obs"]

        action = self.select_action(
            obs=obs,
            legal_actions=legal_actions,
            epsilon=0.0,
        )

        return action, {}

    def _max_next_q(self, next_q_values, next_legal_actions):
        """
        Compute max_a' Q_target(s', a') over legal next actions.

        Used for standard DQN/Q-learning target.
        """
        max_next_q = []

        for i, legal in enumerate(next_legal_actions):
            if len(legal) == 0:
                max_next_q.append(torch.tensor(0.0, device=self.device))
            else:
                masked = torch.full_like(next_q_values[i], -1e9)
                masked[legal] = next_q_values[i, legal]
                max_next_q.append(masked.max())

        return torch.stack(max_next_q)

    def _expected_softmax_next_q(self, next_q_values, next_legal_actions):
        """
        Compute E_{a' ~ softmax(Q_target(s') / temperature)}
                [Q_target(s', a')]

        probably better than sampling from softmax for lookahead
        """
        expected_next_q = []

        for i, legal in enumerate(next_legal_actions):
            if len(legal) == 0:
                expected_next_q.append(torch.tensor(0.0, device=self.device))
                continue

            legal_actions_t = torch.tensor(
                legal,
                dtype=torch.long,
                device=self.device,
            )

            legal_q_values = next_q_values[i, legal_actions_t]

            if self.temperature <= 1e-8:
                expected_next_q.append(legal_q_values.max())
                continue

            probs = F.softmax(legal_q_values / self.temperature, dim=0)
            expected_value = torch.sum(probs * legal_q_values)

            expected_next_q.append(expected_value)

        return torch.stack(expected_next_q)

    def train_step(self):
        """
        One DQN gradient update from replay memory
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.q_network(obs)

        # Extract Q(s, a_taken) from the vector Q(s, .).
        chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_obs)

            if self.target_pol == "greedy":
                next_values = self._max_next_q(
                    next_q_values=next_q_values,
                    next_legal_actions=batch["next_legal_actions"],
                )

            elif self.target_pol == "softmax":
                next_values = self._expected_softmax_next_q(
                    next_q_values=next_q_values,
                    next_legal_actions=batch["next_legal_actions"],
                )

            else:
                raise ValueError(f"Unknown target_pol: {self.target_pol}")

            target = rewards + self.gamma * (1.0 - dones) * next_values

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
                "behavior_pol": self.behavior_pol,
                "target_pol": self.target_pol,
                "temperature": self.temperature,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.train_steps = checkpoint.get("train_steps", 0)

        self.behavior_pol = checkpoint.get("behavior_pol", self.behavior_pol)
        self.target_pol = checkpoint.get("target_pol", self.target_pol)
        self.temperature = checkpoint.get("temperature", self.temperature)