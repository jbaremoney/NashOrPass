class SimplifiedLeducMDP:
    def __init__(self):
        self.action_facing_states = [
            # can't be facing check, open check treated as none
            'none',  # First to act or checked to
            'utg_call',  # Facing initial bet/open
            'call',
            'raise',  # Facing a raise (can reraise)
            'reraise',  # Facing reraise (CAPPED - fold/call only)
        ]
        self.gamma = 1 # finite hands, reward based on outcome of hand


    def reward(self, state, start_bank=None, end_bank=None):
        if state.end_hand:
            return end_bank - start_bank
        else:
            return 0

    def dist_next_states(self, state, pot):
        # get the probability distribution over all possible next states

        pass