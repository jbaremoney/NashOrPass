from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import ACTION_NAME_TO_ID


class TightAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        legal = set(state["legal_actions"].keys())

        check = ACTION_NAME_TO_ID["check"]
        fold = ACTION_NAME_TO_ID["fold"]
        call = ACTION_NAME_TO_ID["call"]

        if check in legal:
            return check
        if fold in legal:
            return fold
        if call in legal:
            return call

        return list(legal)[0]

    def eval_step(self, state):
        action = self.step(state)
        return action, {}