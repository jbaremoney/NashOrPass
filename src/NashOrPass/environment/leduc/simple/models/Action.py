class Action:
    def __init__(self, type: str, player: int = None, amount_raised: int = None, total_amount: int = None):
        self.type = type
        self.player = player
        self.amount_raised = amount_raised
        self.total_amount = total_amount