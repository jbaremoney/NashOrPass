from NashOrPass.limit_holdem.poker_theory.equity import call_is_profitable

from NashOrPass.limit_holdem.envs.rlcard_limit_holdem import ACTION_NAME_TO_ID


class PotOddsAgent:
    def __init__(self):
        self.use_raw = False
    def step(self, state):
        legal = set(state["legal_actions"].keys())

        call = ACTION_NAME_TO_ID["call"]
        check = ACTION_NAME_TO_ID["check"]
        fold = ACTION_NAME_TO_ID["fold"]



        if call in legal and call_is_profitable():

            return call
        if check in legal:
            return check
        return fold

    def eval_step(self, state):
        action = self.step(state)
        return action, {}