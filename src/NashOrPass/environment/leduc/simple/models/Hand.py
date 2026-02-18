from src.NashOrPass.environment.leduc.simple.models.State import State, MDPState
from src.NashOrPass.environment.leduc.simple.models.Action import Action
from src.NashOrPass.environment.leduc.simple.models.BettingRound import BettingRound
from src.NashOrPass.environment.leduc.simple.models.Deck import Deck
from src.NashOrPass.environment.leduc.simple.utils.showdown import showdown

class Hand:
    def __init__(self, bb_amnt, players, deck: Deck, btn, ante: int = None, rules:dict = None):
        self.ante = ante
        self.players = players
        self.hands = [None, None]
        self.btn = btn # index 0 or 1 lined up with players/hands
        print(f"BB: {1-btn}")
        print(f"BTN: {btn}")
        self.rules = rules # ie max bat, max num raises
        self.rounds = [] # list of betting rounds, rounds have list of actions
        self.round = BettingRound("preflop") # initialize to preflop betting round
        self.bb_amnt = bb_amnt
        self.action_to = btn # preflop, btn acts first
        self.flop = None
        self.deck = deck # need state of deck to persist
        self.pot = 0
        self.dealt = False

    def get_mdp_state(self, hero_idx=0):
        return MDPState(self.hands[hero_idx], self.hands[1-hero_idx], self.round.stage,
                        self.flop, self.round.actions[-1], 'btn' if hero_idx == self.btn else 'bb',
                        self.bb_amnt, self.action_to, self.pot, 'check' in [a.type for a in self.round.actions])

    # randomly assign each player one card from 6 card deck
    def deal(self):
        # hands are index aligned, ie players[0]'s hand is in hands[0]
        if self.round.stage == "preflop":
            print("Dealing preflop")
            card1 = self.deck.deal()
            print(f"bb card: {card1}")
            card2 = self.deck.deal()
            print(f"btn card: {card2}")
            print("--------------")
            self.dealt = True

            # not button gets first card
            self.hands[1-self.btn] = card1
            self.hands[self.btn] = card2
        else:
            # not preflop means flop
            self.flop = self.deck.deal()

    def set_flop(self):
        card = self.deck.deal()
        self.flop = card

    def get_poss_actions(self):
        acts = self.round.actions

        if len(acts) == 0:
            return ['check', 'bet']

        last_action = acts[-1].type

        if last_action == 'check':
            return ['check', 'bet']
        elif last_action == 'bet':
            return ['fold', 'call', 'raise']

        elif last_action == 'raise':
            # can always reraise the raise
            return ['fold', 'call', 'reraise']

        elif last_action == 'reraise':
            # can't raise again
            return ['fold', 'call']

        return []

    def build_hand_state(self) -> State:
        at = self.action_to
        hero_card = self.hands[at][0] if self.dealt else None  # Just rank: 'J', 'Q', 'K'
        hero_posn = 'btn' if self.btn == at else 'bb'
        flop_card = self.flop[0] if self.flop else None

        # Determine action facing
        if len(self.round.actions) == 0:
            action_facing = 'none'
        else:
            last_action = self.round.actions[-1].type
            if last_action == 'check':
                action_facing = 'none'  # Checked to us
            else:
                action_facing = last_action  # 'bet', 'raise', or 'reraise'

        return State(hero_card=hero_card, round_stage=self.round.stage,
                     flop_card=flop_card, action_facing=action_facing,
                     position=hero_posn)
    def run_betting_round(self, loud=False):
        # TODO move this logic to BettingRound
        # pot and stack updates are iterative and upon action taken.
        # an open and a call always is of size 1
        # a raise and a rereaise is always of size 2

        if len(self.rounds) == 0:
            self.round = BettingRound("preflop")
            self.action_to = self.btn
        else:
            self.round = BettingRound("postflop") # only two betting rounds
            self.action_to = 1-self.btn
        betting = True
        if (self.players[0].get_bank() == 0 or self.players[1].get_bank() == 0):
            betting = False

        while betting:

            to_act = self.players[self.action_to]

            hand_state = self.build_hand_state()
            if loud:
                print(f"TO ACT: {'btn' if to_act.id == self.btn else 'bb'}")
                print(f"STATE FACING: {hand_state.to_tuple()}")


            # action will be dict
            # blocking?
            action = to_act.act(hand_state, self.bb_amnt)
            if loud:
                print(f"ACTION TAKEN: {action.type}")

            if action.type == "fold":

                self.round.actions.append(Action("fold", self.action_to))
                betting = False

            elif action.type == "check":
                self.round.actions.append(Action("check", self.action_to))
                # this won't break preflop since actions[-2] won't be check
                if (len(self.round.actions) >=2 and self.round.actions[-2].type == "check"):
                    # check back, end  betting round
                    betting = False

                else:
                    # opening check, move to next player
                    self.action_to = 1 if self.action_to == 0 else 0

            elif action.type == "bet":
                self.round.actions.append(Action("bet", self.action_to, total_amount=1 * self.bb_amnt))

                self.pot += 1*self.bb_amnt
                # move action
                self.action_to = 1 if self.action_to == 0 else 0

            elif action.type == "utg_call":
                # special case of call, preflop utg call gives bb option
                self.round.actions.append(Action("call", self.action_to, total_amount=1 * self.bb_amnt))

                self.pot += 1 * self.bb_amnt
                # move action
                self.action_to = 1 if self.action_to == 0 else 0

            elif action.type == "call":
                self.round.actions.append(Action("call", self.action_to, total_amount=1 * self.bb_amnt))

                self.pot += 1*self.bb_amnt
                # call always ends betting round
                betting = False

            else: # RAISE OR RERAISE
                self.round.actions.append(Action(action.type, self.action_to, total_amount=action.total_amount))
                self.pot += 2*self.bb_amnt

                # move action
                self.action_to = 1 if self.action_to == 0 else 0

            if (self.players[0].get_bank() == 0 or self.players[1].get_bank() == 0 ):
                betting = False

        self.rounds.append(self.round)

    def post_antes(self):
        # TODO handle small stack
        if self.ante is not None:
            for player in self.players:
                player.post_ante()
                self.pot += self.ante

    def post_bb(self):
        # TODO handle small stack
        self.players[1-self.btn].post_bb(self.bb_amnt)
        self.pot += self.bb_amnt

    def run(self, loud=False):
        # assign each player a card
        self.post_antes() # adds to pot
        self.post_bb() # adds to pot

        self.deal()

        # run betting round
        self.run_betting_round(loud)

        # check if last action was a fold, ends hand
        if self.rounds[-1].actions[-1].type == "fold":
            # end hand
            folder_idx = self.rounds[-1].actions[-1].player
            self.players[1-folder_idx].update_bank(self.pot, True)
            winner = 1 - folder_idx

        else: #proceed, no one folded
            # deal the flop
            # updates hand state
            self.round = BettingRound("postflop")

            self.deal()

            print(f"POSTFLOP DEALT")

            self.run_betting_round(loud)

            if self.rounds[-1].actions[-1].type != "fold":
                # showdown
                # if return 0, player 1 wins, if return 1, player 2 wins, if return 2, chop
                res = showdown(self.hands[0], self.hands[1], self.flop)
                winner = res
                if res == 0:

                    self.players[0].update_bank(self.pot, True)
                elif res == 1:

                    self.players[1].update_bank(self.pot, True)

                else: # chop
                    self.players[0].update_bank(int(.5*self.pot), True)
                    self.players[1].update_bank(int(.5*self.pot), True)
            else:
                # handle fold
                folder_idx = self.rounds[-1].actions[-1].player
                self.players[1 - folder_idx].update_bank(self.pot,True )
                winner = 1 - folder_idx


        return {"win": winner, "p1_balance": self.players[0].get_bank(), "p2_balance": self.players[1].get_bank() }

