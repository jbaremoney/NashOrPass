class SimplifiedLeducMDP:
    def __init__(self):
        self.action_facing_states = [
            'opening',  # First to act or checked to
            'facing_bet',  # Facing initial bet/open
            'facing_raise',  # Facing a raise (can reraise)
            'facing_reraise',  # Facing reraise (CAPPED - fold/call only)
        ]

    def get_legal_actions(self, state):
        """Return legal actions based on what you're facing"""
        card, round_stage, flop, action_facing, position = state

        if action_facing == 'opening':
            # Can check or open bet
            return ['check', 'bet']

        elif action_facing == 'facing_bet':
            # Facing initial bet - can fold, call, or raise it
            return ['fold', 'call', 'raise']

        elif action_facing == 'facing_raise':
            # Facing a raise - can fold, call, or reraise
            return ['fold', 'call', 'reraise']

        elif action_facing == 'facing_reraise':
            # Facing reraise - CAPPED, can only fold or call
            return ['fold', 'call']

        return []