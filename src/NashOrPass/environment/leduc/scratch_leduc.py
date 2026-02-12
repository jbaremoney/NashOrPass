import random

from pydantic import BaseModel
from typing import Optional, List
from simulation.leduc.models.Deck import Deck
from simulation.leduc.utils.showdown import showdown
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
    def __init__(self, id: int, state: PlayerState, policy="random"):
        self.state = state
        self.policy = Policy(policy)
        self.id = id # either 0 or 1

    def update_balance(self, amnt: int, plus: bool=False):
        """

        :param plus: True if adding, False if subtracting
        :return:
        """

    #choose action from provided list
    def act(self, hand_state, actions):
        choice = self.policy.apply(hand_state, actions)[0] # make sure this is a dict
        if choice == "fold":
            action = Action("fold", self.id)
        elif choice == "check":
            action = Action("check", self.id)
        elif choice == "call":
            self.update_balance(choice["total_amount"])
            action = Action("call", self.id, 0, choice["total_amount"])
        else:
            # raise
            self.update_balance(choice["total_amount"])
            action = Action("raise", self.id, choice["raise_amount"], choice["total_amount"])

        return action

    def post_ante(self, ante):
        self.update_balance(ante)

    def post_bb(self, bb):
        self.update_balance(bb)


class Hand:
    def __init__(self, bb_amnt, players, btn, ante: int = None, rules:dict = None):
        self.ante = ante
        self.players = players
        self.hands = None
        self.btn = btn # index 0 or 1 lined up with players/hands
        self.rules = rules # ie max bat, max num raises
        self.rounds = [] # list of betting rounds, rounds have list of actions
        self.round = BettingRound("PREFLOP") # initialize to preflop betting round
        self.bb_amnt = bb_amnt
        self.action_to = 1-btn # index in list
        self.flop = None
        self.deck = Deck() # need state of deck to persist
        self.pot = 0

    # randomly assign each player one card from 6 card deck
    def deal(self):
        # hands are index aligned, ie players[0]'s hand is in hands[0]
        if self.round == "PREFLOP":
            card1 = self.deck.deal()
            card2 = self.deck.deal()

            # not button gets first card
            self.hands[1-self.btn] = card1
            self.hands[self.btn] = card2
        else:
            # not preflop means flop
            self.flop = self.deck.deal()

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


    def build_hand_state(self):
        """
        build the hand state that will be passed to action player before they make their bet
        Returns:
            hand_state (dict)
        """
        # probably should make this have arguments

        at = self.action_to
        opponent_stack = self.players[1 - at ].state.bank
        hand_dealt = self.hands[at]
        card_flopped = self.flop
        rd = self.round

        #small list this is fine
        past_actions = [actn for actn in rd.actions for rd in self.rounds]

        hand_state = {"hand_dealt": hand_dealt,
                      "pot": self.pot,
                      "round": rd,
                      "card_flopped": card_flopped,
                      "opponent_bank": opponent_stack,
                      "past_actions": past_actions
                      }
        return hand_state

    def run_betting_round(self):
        # TODO move this logic to BettingRound
        if len(self.rounds) == 0:
            round = BettingRound("preflop")
        else:
            round = BettingRound("postflop") # only two betting rounds
        betting = True
        while betting:
            to_act = self.players[self.action_to]
            actions = self.get_poss_actions()
            hand_state = self.build_hand_state()

            # action will be dict
            # blocking?
            action = to_act.act(hand_state, actions)

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

                self.pot += action["total_amount"]
                # call always ends betting round
                betting = False

            else: # RAISE
                # raise never ends betting round
                round.actions.append(Action("raise", self.action_to, action["raise_amount"], action["total_amount"]))

                #pot update
                self.pot += action["total_amount"]

                #action moves
                self.action_to = 1 if self.action_to == 0 else 0

        self.rounds.append(round)

    def post_antes(self):
        # TODO handle small stack
        if self.ante is not None:
            for player in self.players:
                player.post_ante()
                self.pot += 2*self.ante

    def post_bb(self):
        # TODO handle small stack
        self.players[1- self.btn].post_bb()
        self.pot += self.bb_amnt

    def run(self):
        # assign each player a card
        self.post_antes() # adds to pot
        self.post_bb() # adds to pot

        self.deal()

        # run betting round
        self.run_betting_round()

        # check if last action was a fold
        if self.rounds[-1].actions[-1].type == "fold":
            # end hand
            pass

        else:
            # deal the flop
            # updates hand state
            self.deal()

            self.run_betting_round()

            if self.rounds[-1].actions[-1].type != "fold":
                # showdown
                # if return 0, player 1 wins, if return 1, player 2 wins, if return 2, chop
                res = showdown(self.hands[0], self.hands[1], self.flop)


    def stop(self):
        pass
