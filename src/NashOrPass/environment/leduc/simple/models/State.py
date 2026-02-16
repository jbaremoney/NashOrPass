class State:
    def __init__(self, hero_card, round_stage, flop_card, action_facing, position):
        self.hero_card = hero_card  # 'J', 'Q', or 'K'
        self.round_stage = round_stage  # 'preflop' or 'postflop'
        self.flop_card = flop_card  # 'J', 'Q', 'K', or None
        self.action_facing = action_facing  # 'opening', 'facing_bet', 'facing_raise'
        self.position = position  # 'btn' or 'bb'

    def to_tuple(self):
        """Convert to hashable tuple for use as dict key"""
        return (self.hero_card, self.round_stage, self.flop_card,
                self.action_facing, self.position)

    def get_legal_actions(self):
        """Return legal actions based on what you're facing"""
        action_facing = self.action_facing

        if action_facing == 'none':
            if self.round_stage == 'postflop':
                return ['check', 'bet']
            else:
                return ['fold', 'call', 'raise']
        elif action_facing == 'bet':  # CHANGE FROM 'open'
            return ['fold', 'call', 'raise']
        elif action_facing == 'raise':
            return ['fold', 'call', 'reraise']
        elif action_facing == 'reraise':
            return ['fold', 'call']

        return []