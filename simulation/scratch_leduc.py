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

class BettingRound(BaseModel):
    actions: List[Action]


class Player:
    def __init__(self, state):
        self.state = state

    #choose action from provided list
    def act(self, actions):
        pass


class Hand:
    def __init__(self, ante, players, btn, rules:dict):
        self.ante = ante
        self.players = players
        self.btn = btn
        self.rules = rules # ie max bat, max num raises
        self.rounds = [] # list of betting rounds, rounds have list of actions
        self.round = "PREFLOP" # betting round ie preflop, postflop


        self.action_to = 1-btn # index in list

    # randomly assign each player one card from 6 card deck
    def deal(self):
        pass

    def run_betting_round(self):
        round = BettingRound()
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
    def request_action(self):
        poss_actions = self.get_poss_actions()
        player = self.action_to


    def stop(self):
        pass
