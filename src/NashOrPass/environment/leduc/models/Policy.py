import random
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

