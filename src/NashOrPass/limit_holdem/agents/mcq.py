import random
import torch
import torch.nn.functional as F

from NashOrPass.limit_holdem.networks.mlp_qnet import MLPQNetwork
from NashOrPass.limit_holdem.buffers.replay_buffer import ReplayBuffer


class MCQAgent:
    """
    Monte Carlo Q agent.

    This is NOT standard DQN.

    Instead of using a bootstrapped target

        target = r + gamma * max_a Q(s', a)

    this agent directly regresses Q(s, a) toward the realized final payoff
    from the episode/hand.

    For terminal-reward poker hands, this means every transition from a hand
    can receive the same final payoff as its training target.

    Example target:

        Q(obs_t, action_t) -> final_payoff_of_hand

    This is useful as a diagnostic baseline:
        - If this learns but DQN does not, bootstrapping is likely the issue.
        - If this also fails, the bug is probably in trajectories, payoffs,
          action masking, legal actions, or evaluation.
    """

    def __init__(
        self,
        state_dim=72,
        num_actions=4,
        hidden_dim=256,
        gamma=1.0,
        lr=1e-3,
        buffer_capacity=50_000,
        batch_size=64,
        device=None,
    ):
        self.use_raw = False

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_steps = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = MLPQNetwork(
            state_dim,
            hidden_dim,
            num_actions,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=lr,
        )

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def _masked_argmax(self, q_values, legal_actions):
        """
        Return the legal action with the largest Q-value.
        """
        masked = torch.full_like(q_values, -1e9)
        masked[legal_actions] = q_values[legal_actions]
        return int(torch.argmax(masked).item())

    def select_action(self, obs, legal_actions, epsilon=0.0):
        """
        Epsilon-greedy action selection over legal actions.
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

    def step(self, state):
        """
        RLCard-compatible action method.

        During evaluation, this is greedy.
        During training, you can wrap the agent and pass epsilon externally.
        """
        legal_actions = list(state["legal_actions"].keys())
        obs = state["obs"]

        return self.select_action(
            obs=obs,
            legal_actions=legal_actions,
            epsilon=0.0,
        )

    def eval_step(self, state):
        action = self.step(state)
        return action, {}

    def train_step(self):
        """
        Monte Carlo Q update.

        The replay buffer should already contain rewards equal to realized
        final hand returns.

        So the target is simply:

            target = reward

        There is no next-state bootstrap term.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        obs = torch.tensor(
            batch["obs"],
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            batch["actions"],
            dtype=torch.long,
            device=self.device,
        )

        returns = torch.tensor(
            batch["rewards"],
            dtype=torch.float32,
            device=self.device,
        )

        q_values = self.q_network(obs)

        chosen_q = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # Monte Carlo target: realized final payoff.
        target = returns

        loss = F.mse_loss(chosen_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.train_steps = checkpoint.get("train_steps", 0)