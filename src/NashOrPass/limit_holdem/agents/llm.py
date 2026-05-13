import random

class LLMAgent:
    def __init__(self, model="chatgpt", hist_capacity=0):
        self.hist_capacity=hist_capacity
        self.model = model
        self.use_raw = False

    def step(self, state):
        legal_actions = list(state["legal_actions"].keys())
        return random.choice(legal_actions)

    def eval_step(self, state):
        action = self.step(state)
        return action, {}