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
                return ['fold', 'utg_call', 'raise']
        elif action_facing == 'utg_call':
            return ['check', 'raise']
        elif action_facing == 'bet':  # CHANGE FROM 'open'
            return ['fold', 'call', 'raise']
        elif action_facing == 'raise':
            return ['fold', 'call', 'reraise']
        elif action_facing == 'reraise':
            return ['fold', 'call']

        return []

class MDPState(State):
    def __init__(self, hero_card, villain_card, round_stage, flop_card, action_facing, position, bb_amnt, to_act, pot, checked_alr):
        super(MDPState, self).__init__(hero_card, round_stage, flop_card, action_facing, position)
        self.action_facing = action_facing.type if hasattr(action_facing, "type") else action_facing
        self.villain_card = villain_card
        self.bb_amnt = bb_amnt
        self.to_act = to_act
        self.pot = pot
        self.checked_alr = checked_alr
    def to_tuple(self):
        return (self.hero_card, self.villain_card, self.round_stage, self.flop_card,
                self.action_facing, self.position, self.bb_amnt, self.to_act, self.pot, self.checked_alr)