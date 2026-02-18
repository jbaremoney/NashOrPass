from src.NashOrPass.environment.leduc.simple.models.Policy import Policy
from src.NashOrPass.environment.leduc.simple.models.State import MDPState
from src.NashOrPass.environment.leduc.simple.models.Deck import Deck
from copy import deepcopy


class LeducSimpleMDP:
    def __init__(self, initial_state: MDPState):
        # FIELDS: hero_card, villain_card, round_stage, flop_card, action_facing, position, bb_amnt, to_act, pot, checked_alr
        self.state = initial_state

    def legal_actions(self):
        return self.state.get_legal_actions()

    @staticmethod
    def other(i):
        return 1 - i

    @staticmethod
    def legal_actions_from_mdp(s: MDPState):
        if s.action_facing == 'none':
            if s.round_stage == 'postflop':
                return ['check', 'bet']
            else:
                return ['fold', 'utg_call', 'raise']
        elif s.action_facing == 'utg_call':
            return ['check', 'raise']
        elif s.action_facing == 'bet':
            return ['fold', 'call', 'raise']
        elif s.action_facing == 'raise':
            return ['fold', 'call', 'reraise']
        elif s.action_facing == 'reraise':
            return ['fold', 'call']
        return []

    @staticmethod
    def apply_action(s: MDPState, a: str):
        """
        returns new state after applying action specified.

        Note that if the action results in the end of betting round or hand it returns as such, STRICTLY DETERMINISTIC

        """
        def action_to_step(s_p: MDPState, a):
            """this is called any time move to other player to act, steps action to and pos'n forward.
            also changes action facing"""
            s_p.to_act = 1 - s_p.to_act
            # change action facing
            s_p.action_facing = a
            # move to other position 'bb' or 'btn' ?
            s_p.position = 'bb' if s_p.position == 'btn' else 'btn'

            return s_p


        s_p = deepcopy(s)
        # handle preflop states
        if s_p.round_stage == 'preflop':
            # TODO handle folding/terminal states
            if s.action_facing == 'none':
                # opening preflop action
                if a == 'utg_call':
                    # add 1 to pot
                    s_p.pot += 1
                    # step forward
                    s_p = action_to_step(s_p, a)

                elif a == 'fold':
                    ...

                elif a == 'raise':
                    s_p.pot += 2
                    # step forward
                    s_p = action_to_step(s_p, a)

                else:
                    raise ValueError("Action taken when facing 'none' preflop INVALID")

            elif s_p.action_facing == 'utg_call':
                # on the button, option
                if a == 'check':
                    # ends betting round, flops card. action stays bb
                    ...

                elif a == 'raise':
                    s_p.pot += 2
                    # move action
                    s_p = action_to_step(s_p, a)

                else:
                    raise ValueError()


        # return new state
        return s_p