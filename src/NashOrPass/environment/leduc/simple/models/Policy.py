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
        print(f"_weighted_choice called with dist {dist}")
        print(f"dist type: {type(dist)}")

        r = random.random()
        cum = 0.0
        for a, p in dist.items():
            cum += p
            if r <= cum:
                print(f"SAMPLED ACTION: {a}")
                return a
        # fallback for float error

        return list(dist.keys())[-1]

    def _bucket_postflop(self, state: State):
        """
        Uses actor's private card (state.hero_card) and public flop card (state.flop_card)
        """
        op = state.hero_card[0]          # 'J','Q','K'
        flop = state.flop_card           # 'J','Q','K' or None

        # classifying hands postflop
        if flop is not None and op == flop:
            return "PAIR"
        if op == "K":
            return "HIGH"
        if op == "Q":
            return "MID"
        return "LOW"

    def dummy_dist(self, leg_actions, state: State) -> dict:
        """
        Return a probability distribution over legal actions.

        Output format:
            {"action_name": probability, ...}

        This is for DP, so it returns probabilities, not a sampled action.
        """

        def uniform():
            n = len(leg_actions)
            if n == 0:
                return {}
            return {a: 1.0 / n for a in leg_actions}

        def normalize(dist):
            # keep only legal actions
            dist = {a: p for a, p in dist.items() if a in leg_actions}

            total = sum(dist.values())
            if total <= 0:
                return uniform()

            return {a: p / total for a, p in dist.items()}

        if self.type == "random" or state is None:
            return uniform()

        if self.type != "standard":
            raise ValueError(f"Policy type {self.type} not implemented for dummy_dist")

        card = state.hero_card[0] if state.hero_card else None
        facing = state.action_facing
        stage = state.round_stage

        # --------------------------------------------------
        # PREFLOP OPENING: fold / utg_call / raise
        # --------------------------------------------------
        if stage == "preflop" and facing == "none":
            if card == "K":
                return normalize({"fold": 0.00, "utg_call": 0.45, "raise": 0.55})
            elif card == "Q":
                return normalize({"fold": 0.05, "utg_call": 0.75, "raise": 0.20})
            else:  # J
                return normalize({"fold": 0.15, "utg_call": 0.75, "raise": 0.10})

        # --------------------------------------------------
        # PREFLOP FACING UTG_CALL: check / raise
        # --------------------------------------------------
        if stage == "preflop" and facing == "utg_call":
            if card == "K":
                return normalize({"check": 0.35, "raise": 0.65})
            elif card == "Q":
                return normalize({"check": 0.75, "raise": 0.25})
            else:  # J
                return normalize({"check": 0.90, "raise": 0.10})

        # --------------------------------------------------
        # POSTFLOP OPENING: check / bet
        # --------------------------------------------------
        if stage == "postflop" and facing == "none":
            b = self._bucket_postflop(state)

            if b == "PAIR":
                return normalize({"check": 0.10, "bet": 0.90})
            elif b == "HIGH":
                return normalize({"check": 0.40, "bet": 0.60})
            elif b == "MID":
                return normalize({"check": 0.65, "bet": 0.35})
            else:  # LOW
                return normalize({"check": 0.80, "bet": 0.20})

        # --------------------------------------------------
        # POSTFLOP AFTER CHECK: check / bet
        # --------------------------------------------------
        if stage == "postflop" and facing == "check":
            b = self._bucket_postflop(state)

            if b == "PAIR":
                return normalize({"check": 0.20, "bet": 0.80})
            elif b == "HIGH":
                return normalize({"check": 0.45, "bet": 0.55})
            elif b == "MID":
                return normalize({"check": 0.70, "bet": 0.30})
            else:  # LOW
                return normalize({"check": 0.85, "bet": 0.15})

        # --------------------------------------------------
        # FACING BET: fold / call / raise
        # --------------------------------------------------
        if facing == "bet":
            b = self._bucket_postflop(state)

            if b == "PAIR":
                return normalize({"fold": 0.00, "call": 0.35, "raise": 0.65})
            elif b == "HIGH":
                return normalize({"fold": 0.05, "call": 0.70, "raise": 0.25})
            elif b == "MID":
                return normalize({"fold": 0.20, "call": 0.70, "raise": 0.10})
            else:
                return normalize({"fold": 0.45, "call": 0.50, "raise": 0.05})

        # --------------------------------------------------
        # FACING RAISE: fold / call / reraise
        # Works preflop and postflop.
        # --------------------------------------------------
        if facing == "raise":
            # Preflop: use card strength only
            if stage == "preflop":
                if card == "K":
                    return normalize({"fold": 0.00, "call": 0.35, "reraise": 0.65})
                elif card == "Q":
                    return normalize({"fold": 0.15, "call": 0.75, "reraise": 0.10})
                else:  # J
                    return normalize({"fold": 0.45, "call": 0.55, "reraise": 0.00})

            # Postflop: use bucket
            b = self._bucket_postflop(state)

            if b == "PAIR":
                return normalize({"fold": 0.00, "call": 0.30, "reraise": 0.70})
            elif b == "HIGH":
                return normalize({"fold": 0.10, "call": 0.75, "reraise": 0.15})
            elif b == "MID":
                return normalize({"fold": 0.30, "call": 0.65, "reraise": 0.05})
            else:
                return normalize({"fold": 0.60, "call": 0.40, "reraise": 0.00})

        # --------------------------------------------------
        # FACING RERAISE: fold / call
        # Works preflop and postflop.
        # --------------------------------------------------
        if facing == "reraise":
            if stage == "preflop":
                if card == "K":
                    return normalize({"fold": 0.05, "call": 0.95})
                elif card == "Q":
                    return normalize({"fold": 0.35, "call": 0.65})
                else:  # J
                    return normalize({"fold": 0.75, "call": 0.25})

            b = self._bucket_postflop(state)

            if b == "PAIR":
                return normalize({"fold": 0.00, "call": 1.00})
            elif b == "HIGH":
                return normalize({"fold": 0.25, "call": 0.75})
            elif b == "MID":
                return normalize({"fold": 0.55, "call": 0.45})
            else:
                return normalize({"fold": 0.80, "call": 0.20})

        return uniform()

    def apply(self, legal_actions, state: State = None):
        print(f"APPLYING POLICY to legal actions: {legal_actions}")

        if self.type == "random":
            return random.sample(legal_actions, 1)[0]

        elif self.type == "standard":
            # safety: if no state, default random
            if state is None:
                raise ValueError("State is None passed to villain policy")

            dist = self.dummy_dist(legal_actions, state)
            return self._weighted_choice(dist)


        elif self.type == "strict_value":
            # placeholder: could be deterministic thresholds (e.g., always raise with K, etc.)
            return random.sample(legal_actions, 1)[0]

        # default
        return random.sample(legal_actions, 1)[0]
