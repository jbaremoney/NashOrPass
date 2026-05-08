from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, obs, action, reward, next_obs, done, legal_actions, next_legal_actions):
        self.buffer.append((
            np.array(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            bool(done),
            list(legal_actions),
            list(next_legal_actions),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        obs, actions, rewards, next_obs, dones, legal_actions, next_legal_actions = zip(*batch)

        return {
            "obs": np.stack(obs),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_obs": np.stack(next_obs),
            "dones": np.array(dones, dtype=np.float32),
            "legal_actions": list(legal_actions),
            "next_legal_actions": list(next_legal_actions),
        }

    def __len__(self):
        return len(self.buffer)