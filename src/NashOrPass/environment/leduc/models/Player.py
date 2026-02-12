from src.NashOrPass.environment.leduc.models.Action import Action
from src.NashOrPass.environment.leduc.models.Policy import Policy

class Player:
    def __init__(self, id: int, bank: int, policy="random"):
        self.policy = Policy(policy)
        self.id = id # either 0 or 1
        self.bank = bank

    def update_balance(self, amnt: int, plus: bool=False):
        """
        Updates bank balance
        :param plus: True if adding, False if subtracting
        :return:
        """
        if plus:
            self.bank += amnt
        else:
            self.bank -= amnt

    #choose action from provided list
    # actions is list of Actions
    def act(self, hand_state, actions):
        choice = self.policy.apply(hand_state, actions)[0] # choice will be an Action
        if choice.type == "fold":
            action = Action("fold", self.id)
        elif choice.type == "check":
            action = Action("check", self.id)
        elif choice.type == "call":
            self.update_balance(choice.total_amount)
            action = Action("call", self.id, 0, choice.total_amount)
        else:
            # raise
            self.update_balance(choice["total_amount"])
            action = Action("raise", self.id, choice.raise_amount, choice.total_amount)

        return action

    def post_ante(self, ante):
        self.update_balance(ante)

    def post_bb(self, bb):
        self.update_balance(bb)