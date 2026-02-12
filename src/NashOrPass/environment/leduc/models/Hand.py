from src.NashOrPass.environment.leduc.models.Action import Action
from src.NashOrPass.environment.leduc.models.BettingRound import BettingRound
from src.NashOrPass.environment.leduc.models.Deck import Deck
from src.NashOrPass.environment.leduc.utils.showdown import showdown
from src.NashOrPass.environment.leduc.utils.actions import generate_possible_bet_actions


class Hand:
    def __init__(self, bb_amnt, players, deck: Deck, btn, ante: int = None, rules:dict = None):
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
        self.deck = deck # need state of deck to persist
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
            return ["check", "fold", *generate_possible_bet_actions(prev, self.bb_amnt, min(self.players[0].bank,
                                                                                            self.players[1].bank))]

        else:
            # notice prev action will never be call or fold, since 2 players, those end betting round
            if prev.type == "check":
                # effectively same as opening
                return ["check", "fold", *generate_possible_bet_actions(prev, self.bb_amnt, min(self.players[0].bank,
                                                                                            self.players[1].bank))]

            elif prev.type == "raise":
                # can be open or normal raise
                # TODO handle when bank < min raise, or bank < call amount, etc
                return ["call", "fold", *generate_possible_bet_actions(prev, self.bb_amnt, min(self.players[0].bank,
                                                                                            self.players[1].bank))]


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

        # check if last action was a fold, ends hand
        if self.rounds[-1].actions[-1].type == "fold":
            # end hand
            folder_idx = self.rounds[-1].actions[-1].player
            self.players[1-folder_idx].update_bank(True, self.pot)

        else: #proceed
            # deal the flop
            # updates hand state
            self.round = "POSTFLOP"

            self.deal()

            self.run_betting_round()

            if self.rounds[-1].actions[-1].type != "fold":
                # showdown
                # if return 0, player 1 wins, if return 1, player 2 wins, if return 2, chop
                res = showdown(self.hands[0], self.hands[1], self.flop)
                winner = res
                if res == 0:

                    self.players[0].update_bank(True, self.pot)
                elif res == 1:

                    self.players[1].update_bank(True, self.pot)

                else: # chop
                    self.players[0].update_bank(True, int(.5*self.pot))
                    self.players[1].update_bank(True, int(.5*self.pot))
            else:
                # handle fold
                folder_idx = self.rounds[-1].actions[-1].player
                self.players[1 - folder_idx].update_bank(True, self.pot)
                winner = 1 - folder_idx


        return {"win": winner, "p1_balance": self.players[0].get_bank(), "p2_balance": self.players[1].get_bank() }

