from src.NashOrPass.environment.leduc.simple.models.Action import Action
from src.NashOrPass.environment.leduc.simple.models.Policy import Policy
from src.NashOrPass.environment.leduc.simple.models.State import State

class Player:
    def __init__(self, id: int, bank: int, policy="random"):
        self.policy = Policy(policy)
        self.id = id # either 0 or 1
        self.bank = bank

    def update_bank(self, amount, plus=False):  # RENAME from update_balance
        """Updates bank balance"""
        if plus:
            self.bank += amount
        else:
            self.bank -= amount

    def get_bank(self):  # ADD THIS METHOD
        return self.bank

    def act(self, state: State, bb_amnt):
        """Choose action from legal actions"""
        actions = state.get_legal_actions()
        choice = self.policy.apply(actions)  # Returns a string like "fold", "check"

        if choice == "fold":
            return Action("fold", self.id)
        elif choice == "check":
            return Action("check", self.id)
        elif choice == "bet":  # CHANGE FROM "open"
            self.update_bank(1*bb_amnt, plus=False)
            return Action("bet", self.id, amount_raised=1*bb_amnt, total_amount=1*bb_amnt)
        elif choice == "call":
            self.update_bank(1*bb_amnt, plus=False)
            return Action("call", self.id, amount_raised=0, total_amount=1*bb_amnt)
        elif choice == "raise":
            self.update_bank(2*bb_amnt, plus=False)
            return Action("raise", self.id, amount_raised=1*bb_amnt, total_amount=2*bb_amnt)
        elif choice == "reraise":
            self.update_bank(2*bb_amnt, plus=False)
            return Action("reraise", self.id, amount_raised=1*bb_amnt, total_amount=2*bb_amnt)

        return Action("fold", self.id)  # Default fallback

    def post_ante(self, ante=1):  # ADD amount parameter
        self.update_bank(ante, plus=False)

    def post_bb(self, bb=1):  # ADD amount parameter
        self.update_bank(bb, plus=False)