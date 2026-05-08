from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import ACTION_NAME_TO_ID


class AggressiveAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        legal = set(state["legal_actions"].keys())

        raise_action = ACTION_NAME_TO_ID["raise"]
        call = ACTION_NAME_TO_ID["call"]
        check = ACTION_NAME_TO_ID["check"]
        fold = ACTION_NAME_TO_ID["fold"]

        if raise_action in legal:
            return raise_action
        if call in legal:
            return call
        if check in legal:
            return check
        return fold

    def eval_step(self, state):
        action = self.step(state)
        return action, {}