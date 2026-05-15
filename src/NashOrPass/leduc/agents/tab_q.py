import pickle
import random
from collections import defaultdict


class TabularQAgent:
    """
    Tabular Q-learning agent for the custom Leduc MDP.

    This is intentionally simple and interpretable.

    obs_mode:
        "perfect":
            Uses state.to_tuple(), matching the DP policy key.

        "imperfect":
            Uses only hero-observable information.
            If state has villain_card, it is excluded.
    """

    def __init__(
        self,
        actions,
        obs_mode="perfect",
        alpha=0.1,
        gamma=1.0,
        epsilon=0.1,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        seed=None,
    ):
        if obs_mode not in {"perfect", "imperfect"}:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")

        self.actions = list(actions)
        self.obs_mode = obs_mode

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = random.Random(seed)

        # Q maps:
        #   obs_key -> action -> value
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

        self.train_steps = 0

    def obs_key(self, state):
        """
        Convert MDP State into the state representation used by Q-learning.
        """

        if self.obs_mode == "perfect":
            return state.to_tuple()

        # Imperfect/observable poker information.
        #
        # This deliberately excludes villain_card if your State ever has it.
        # It only uses fields hero can reasonably observe.
        return (
            getattr(state, "hero_card", None),
            getattr(state, "round_stage", None),
            getattr(state, "flop_card", None),
            getattr(state, "action_facing", None),
            getattr(state, "hero_position", None),
            getattr(state, "folded_player", None),
        )

    def legal_q_values(self, obs_key, legal_actions):
        return {
            action: self.Q[obs_key][action]
            for action in legal_actions
        }

    def greedy_action(self, state, legal_actions):
        obs = self.obs_key(state)
        q_values = self.legal_q_values(obs, legal_actions)

        max_q = max(q_values.values())

        # Random tie-break among best actions.
        best_actions = [
            action
            for action, q in q_values.items()
            if q == max_q
        ]

        return self.rng.choice(best_actions)

    def select_action(self, state, legal_actions, training=True):
        """
        Epsilon-greedy action selection.
        """

        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(list(legal_actions))

        return self.greedy_action(state, legal_actions)

    def update(self, state, action, reward, next_state, done, next_legal_actions):
        """
        Standard Q-learning update:

            Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]

        where:

            target = r                         if done
            target = r + gamma max_a' Q(s',a') otherwise
        """

        obs = self.obs_key(state)
        old_q = self.Q[obs][action]

        if done:
            target = reward
        else:
            next_obs = self.obs_key(next_state)
            max_next_q = max(
                self.Q[next_obs][a]
                for a in next_legal_actions
            )
            target = reward + self.gamma * max_next_q

        td_error = target - old_q
        self.Q[obs][action] = old_q + self.alpha * td_error

        self.train_steps += 1

        return td_error

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def policy_dict(self, reachable_states, legal_action_fn):
        """
        Convert learned Q table into a deterministic greedy policy.

        Returns:
            {state.to_tuple(): greedy_action}
        """
        policy = {}

        for state in reachable_states:
            legal_actions = legal_action_fn(state)
            if not legal_actions:
                continue

            policy[state.to_tuple()] = self.greedy_action(
                state,
                legal_actions,
            )

        return policy

    def save(self, path):
        payload = {
            "Q": dict(self.Q),
            "actions": self.actions,
            "obs_mode": self.obs_mode,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "train_steps": self.train_steps,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path, seed=None):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            actions=payload["actions"],
            obs_mode=payload["obs_mode"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            seed=seed,
        )

        agent.Q = defaultdict(
            lambda: {a: 0.0 for a in agent.actions},
            payload["Q"],
        )

        agent.train_steps = payload.get("train_steps", 0)

        return agent