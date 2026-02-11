import random

from pydantic import BaseModel
from typing import Optional, List
# truth player state
class PlayerState:
    def __init__(self, bank):
        self.bank = bank

    def set_bank(self, amnt):
        self.bank = amnt

    def get_bank(self):
        return self.bank

class Action:
    def __init__(self, type: str, player: int, amount_raised: int = None, total_amount: int = None):
        self.type = type
        self.player = player
        self.amount_raised = amount_raised
        self.total_amount = total_amount

class BettingRound:
    def __init__(self, stage: str):
        self.stage = stage
        self.actions = [] # list of Actions

# change this to PossibleRaise? Since call can just be a string
class PossibleBet:
    def __init__(self, lower, upper, bb):
        # range of values, 1bb increment from lower to upper
        # not sure if this is best way to implement
        self.bet_possibilities = range(lower, upper, bb)


class Policy:
    def __init__(self, type="random"):
        self.type = type
        if self.type == "random":
            self.sd = random.randint(1, 100)
            random.seed(self.sd)

    def apply(self, state, actions):
        if self.type == "random":
            # uses seed defined above
            return random.sample(actions, 1)
        elif self.type == "strict_value":
            # calculate generalized value threshold
            pass



class Player:
    def __init__(self, state, policy="random"):
        self.state = state
        self.policy = Policy(policy)

    #choose action from provided list
    def act(self, hand_state, actions):
        return self.policy.apply(hand_state, actions)


class Hand:
    def __init__(self, ante, bb, players, btn, rules:dict):
        self.ante = ante
        self.players = players
        self.hands = None
        self.btn = btn
        self.rules = rules # ie max bat, max num raises
        self.rounds = [] # list of betting rounds, rounds have list of actions
        self.round = BettingRound("PREFLOP") # initialize to preflop betting round
        self.bb = bb
        self.action_to = 1-btn # index in list
        self.flop = None
        self.deck = None # need state of deck to persist

    # randomly assign each player one card from 6 card deck
    def deal(self):
        # hands are index aligned, ie players[0]'s hand is in hands[0]
        pass

    def set_flop(self):
        card = ""
        self.flop = card

    def get_poss_actions(self):
        # TODO handle rules
        at = self.action_to
        acts = self.round.actions
        prev = acts[-1]

        if len(self.round.actions) == 0:
            # opening
            return ["check", "fold", PossibleBet(self.bb, self.players[at].state.get_bank(), self.bb)]

        else:
            # notice prev action will never be call or fold, since 2 players, those end betting round
            if prev.type == "check":
                # effectively same as opening
                return ["check", "fold", PossibleBet(self.bb, self.players[at].state.get_bank(), self.bb)]

            elif prev.type == "raise":
                # can be open or normal raise
                # TODO handle when bank < min raise, or bank < call amount, etc
                return ["call", "fold", PossibleBet(prev.amount*2, self.players[at].state.get_bank(), self.bb)]



    def run_betting_round(self):
        if len(self.rounds) == 0:
            round = BettingRound("preflop")
        else:
            round = BettingRound("postflop") # only two betting rounds
        betting = True
        while betting:
            to_act = self.players[self.action_to]
            actions = self.get_poss_actions()

            # action will be dict
            action = to_act.act(actions)

            if action["name"] == "fold":

                round.actions.append(Action("fold", self.action_to))
                self.stop() # STOP WHOLE HAND

            elif action["name"] == "check":
                round.actions.append(Action("check", self.action_to))

                if (len(round.actions) >=1 and round.actions[-1].type == "check"):
                    # checked to, ends betting round
                    betting = False

                else:
                    # opening check, move to next player
                    self.action_to = 1 if self.action_to == 0 else 0

            elif action["name"] == "call":
                round.actions.append(Action("call", self.action_to, total_amount=action["total_amount"]))
                # call always ends betting round
                betting = False

            else: # RAISE
                # raise never ends betting round
                round.actions.append(Action("raise", self.action_to, action["raise_amount"], action["total_amount"]))
                self.action_to = 1 if self.action_to == 0 else 0



    # request action from correct player
    def get_player_action(self):
        poss_actions = self.get_poss_actions()
        at = self.action_to
        if self.player[at].human == True:
            pass
        else:
            # pass hand, prev actions, state
            player_hand = self.hands[at]

            action = self.players[at].act()



    def stop(self):
        pass
