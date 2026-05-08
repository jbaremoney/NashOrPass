from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import ACTION_NAME_TO_ID


class CheckCallAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        legal = set(state["legal_actions"].keys())

        check = ACTION_NAME_TO_ID["check"]
        call = ACTION_NAME_TO_ID["call"]
        fold = ACTION_NAME_TO_ID["fold"]

        if check in legal:
            return check
        if call in legal:
            return call
        return fold

    def eval_step(self, state):
        action = self.step(state)
        return action, {}