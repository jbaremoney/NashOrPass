import random
from src.NashOrPass.environment.leduc.simple.models.State import State

class Policy:
    def __init__(self, type="random"):
        self.type = type
        self.sd = random.randint(1, 100)
        random.seed(self.sd)

    def _weighted_choice(self, dist: dict):
        """
        dist: {"action": prob, ...} probs sum to 1 (or close)
        returns one sampled action string
        """
        r = random.random()
        cum = 0.0
        for a, p in dist.items():
            cum += p
            if r <= cum:
                return a
        # fallback for float error
        return list(dist.keys())[-1]

    def _bucket_postflop(self, state: State):
        """
        Uses actor's private card (state.hero_card) and public flop card (state.flop_card)
        """
        op = state.hero_card[0]          # 'J','Q','K'
        flop = state.flop_card           # 'J','Q','K' or None

        if flop is not None and op == flop:
            return "PAIR"
        if op == "K":
            return "HIGH"
        if op == "Q":
            return "MID"
        return "LOW"

    def dist(self, leg_actions, state: State):
        if self.type == "random":
            pass
        elif self.type == "standard":
            # safety: if no state, default random
            if state is None:
                return random.sample(leg_actions, 1)[0]

            # convenience
            card = state.hero_card[0] if state.hero_card else None
            facing = state.action_facing
            stage = state.round_stage

            # actor opening preflop
            if stage == "preflop" and facing == "none":
                if card == "K":
                    dist = {"fold": 0.00, "utg_call": 0.45, "raise": 0.55}
                elif card == "Q":
                    dist = {"fold": 0.05, "utg_call": 0.75, "raise": 0.20}
                else:  # J
                    dist = {"fold": 0.15, "utg_call": 0.75, "raise": 0.10}

                # only keep actions that are actually legal in this spot
                dist = {a: p for a, p in dist.items() if a in leg_actions}
                # renormalize (in case something got filtered)
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- POSTFLOP first action (facing 'none') -----
            # postflop + facing 'none' => ['check','bet']
            if stage == "postflop" and facing == "none":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"check": 0.10, "bet": 0.90}
                elif b == "HIGH":
                    dist = {"check": 0.40, "bet": 0.60}
                elif b == "MID":
                    dist = {"check": 0.65, "bet": 0.35}
                else:  # LOW
                    dist = {"check": 0.80, "bet": 0.20}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- FACING A BET -----
            if facing == "bet" or facing == "utg_call":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"fold": 0.00, "call": 0.35, "raise": 0.65}
                elif b == "HIGH":
                    dist = {"fold": 0.05, "call": 0.70, "raise": 0.25}
                elif b == "MID":
                    dist = {"fold": 0.20, "call": 0.70, "raise": 0.10}
                else:
                    dist = {"fold": 0.45, "call": 0.50, "raise": 0.05}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- FACING A RAISE -----
            if facing == "raise":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"fold": 0.00, "call": 0.30, "reraise": 0.70}
                elif b == "HIGH":
                    dist = {"fold": 0.10, "call": 0.75, "reraise": 0.15}
                elif b == "MID":
                    dist = {"fold": 0.30, "call": 0.65, "reraise": 0.05}
                else:
                    dist = {"fold": 0.60, "call": 0.40, "reraise": 0.00}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- FACING A RERAISE -----
            if facing == "reraise":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"fold": 0.00, "call": 1.00}
                elif b == "HIGH":
                    dist = {"fold": 0.25, "call": 0.75}
                elif b == "MID":
                    dist = {"fold": 0.55, "call": 0.45}
                else:
                    dist = {"fold": 0.80, "call": 0.20}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

    def apply(self, leg_actions, state: State = None):
        if self.type == "random":
            return random.sample(leg_actions, 1)[0]

        elif self.type == "standard":
            # safety: if no state, default random
            if state is None:
                return random.sample(leg_actions, 1)[0]

            # convenience
            card = state.hero_card[0] if state.hero_card else None
            facing = state.action_facing
            stage = state.round_stage

            # actor opening preflop
            if stage == "preflop" and facing == "none":
                if card == "K":
                    dist = {"fold": 0.00, "utg_call": 0.45, "raise": 0.55}
                elif card == "Q":
                    dist = {"fold": 0.05, "utg_call": 0.75, "raise": 0.20}
                else:  # J
                    dist = {"fold": 0.15, "utg_call": 0.75, "raise": 0.10}

                # only keep actions that are actually legal in this spot
                dist = {a: p for a, p in dist.items() if a in leg_actions}
                # renormalize (in case something got filtered)
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- POSTFLOP first action (facing 'none') -----
            # postflop + facing 'none' => ['check','bet']
            if stage == "postflop" and facing == "none":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"check": 0.10, "bet": 0.90}
                elif b == "HIGH":
                    dist = {"check": 0.40, "bet": 0.60}
                elif b == "MID":
                    dist = {"check": 0.65, "bet": 0.35}
                else:  # LOW
                    dist = {"check": 0.80, "bet": 0.20}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- FACING A BET -----
            if facing == "bet" or facing == "utg_call":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"fold": 0.00, "call": 0.35, "raise": 0.65}
                elif b == "HIGH":
                    dist = {"fold": 0.05, "call": 0.70, "raise": 0.25}
                elif b == "MID":
                    dist = {"fold": 0.20, "call": 0.70, "raise": 0.10}
                else:
                    dist = {"fold": 0.45, "call": 0.50, "raise": 0.05}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- FACING A RAISE -----
            if facing == "raise":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"fold": 0.00, "call": 0.30, "reraise": 0.70}
                elif b == "HIGH":
                    dist = {"fold": 0.10, "call": 0.75, "reraise": 0.15}
                elif b == "MID":
                    dist = {"fold": 0.30, "call": 0.65, "reraise": 0.05}
                else:
                    dist = {"fold": 0.60, "call": 0.40, "reraise": 0.00}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # ----- FACING A RERAISE -----
            if facing == "reraise":
                b = self._bucket_postflop(state)
                if b == "PAIR":
                    dist = {"fold": 0.00, "call": 1.00}
                elif b == "HIGH":
                    dist = {"fold": 0.25, "call": 0.75}
                elif b == "MID":
                    dist = {"fold": 0.55, "call": 0.45}
                else:
                    dist = {"fold": 0.80, "call": 0.20}

                dist = {a: p for a, p in dist.items() if a in leg_actions}
                s = sum(dist.values())
                if s == 0:
                    return random.sample(leg_actions, 1)[0]
                dist = {a: p / s for a, p in dist.items()}
                return self._weighted_choice(dist)

            # anything else: fall back
            return random.sample(leg_actions, 1)[0]

        elif self.type == "strict_value":
            # placeholder: could be deterministic thresholds (e.g., always raise with K, etc.)
            return random.sample(leg_actions, 1)[0]

        # default
        return random.sample(leg_actions, 1)[0]
