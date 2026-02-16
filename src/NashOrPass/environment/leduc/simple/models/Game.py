from src.NashOrPass.environment.leduc.simple.models.Hand import Hand
from src.NashOrPass.environment.leduc.simple.models.Deck import Deck
from src.NashOrPass.environment.leduc.simple.models.Player import Player
from typing import List


class Game:
    def __init__(self, players: List[Player], bb_amnt, ante=None, rules=None):
        self.btn = 0
        self.bb_amnt = bb_amnt
        self.players = players
        self.ante = ante
        self.rules = rules
        self.hands = [] # list of Hand objects
        self.current_hand = None

    def play_hand(self):
        deck = Deck()
        new_hand = Hand(self.bb_amnt, self.players, deck, self.btn, self.ante, self.rules)

        out = new_hand.run()

        self.hands.append(new_hand)

        # update state of game after hand is played
        self.players[0].bank = out["p1_balance"]
        self.players[1].bank = out["p2_balance"]
        self.btn = 1 - self.btn # move btn

    def run_game(self, n_hands=None):
        if n_hands is None:
            while (self.players[0].bank > 0 and self.players[1].bank > 0):
                self.play_hand()

        else:
            for i in range(n_hands):
                self.play_hand()
