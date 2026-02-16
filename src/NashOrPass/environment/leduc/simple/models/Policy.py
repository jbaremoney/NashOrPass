import random
class Policy:
    def __init__(self, type="random"):
        self.type = type
        if self.type == "random":
            self.sd = random.randint(1, 100)
            random.seed(self.sd)

    def apply(self, actions):
        if self.type == "random":
            return random.sample(actions, 1)[0]  # ADD [0] to return element, not list
        elif self.type == "strict_value":
            pass

