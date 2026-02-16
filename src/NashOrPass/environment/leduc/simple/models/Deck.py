import random
class Deck:
    # initializing deck every hand is equivalent to shuffling
    def __init__(self):
        # not worried ab efficiency only 6 cards
        self.cards_in = ["Jh", "Jd", "Qh", "Qd", "Kh", "Kd"]
        self.seed = random.randint(1,100)
        random.seed(self.seed)

    def deal(self):
        card = random.sample(self.cards_in, 1)[0]
        self.cards_in.remove(card)
        return card

