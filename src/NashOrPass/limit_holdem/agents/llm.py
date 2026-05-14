import random
import abstract

class LLMAgent:
    def __init__(self, hist_capacity=0):
        self.hist_capacity=hist_capacity
        self.use_raw = False

    def step(self, state, hist=None):
        """choose an action from the state"""
        legal_actions = list(state["legal_actions"].keys())
        prompt = self.build_prompt(state,hist, self.hist_capacity)

        # invoke model with structured output
        # structure should just be ONE OF legal actions
        choice = self.invoke_structured(prompt, legal_actions)

        # maybe check that it's correct?

        return choice


    def invoke_structured(self, prompt, structure):
        pass

    @staticmethod
    def build_prompt(state, hist, hist_cap):
        pass

    def eval_step(self, state):
        action = self.step(state)
        return action, {}


class GPTAgent(LLMAgent):
    def __init__(self, hist_capacity=0):
        super().__init__(hist_capacity)

    def invoke_structured(self, prompt, structure):
        pass


class ClaudeAgent(LLMAgent):
    def __init__(self, hist_capacity=0):
        super().__init__(hist_capacity)

    def invoke_structured(self, prompt, structure):
        pass


class GrokAgent(LLMAgent):
    def __init__(self, hist_capacity=0):
        super().__init__(hist_capacity)

    def invoke_structured(self, prompt, structure):
        pass

