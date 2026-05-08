import random

class RandomAgent:
    def __init__(self, num_actions=None):
        self.num_actions = num_actions
        self.use_raw = False

    def step(self, state):
        legal_actions = list(state["legal_actions"].keys())
        return random.choice(legal_actions)

    def eval_step(self, state):
        action = self.step(state)
        return action, {}