from copy import deepcopy

from NashOrPass.leduc.env.models.Policy import Policy
from NashOrPass.leduc.env.models.State import MDPState
from NashOrPass.leduc.env.utils.showdown import showdown


class LeducSimpleMDP:
    def __init__(self, initial_state: MDPState):
        # FIELDS: hero_card, villain_card, round_stage, flop_card, action_facing, hero_position, bb_amnt, to_act, pot, checked_alr
        # hero_position is fixed, gives absolute pos'n for the hand
        # hero is 0, villain is 1
        self.state = initial_state

    def legal_actions(self):
        return self.state.get_legal_actions()

    @staticmethod
    def other(i):
        return 1 - i

    @staticmethod
    def legal_actions_from_mdp(s: MDPState):
        if s.action_facing == 'none':
            if s.round_stage == 'postflop':
                return ['check', 'bet']
            else:
                return ['fold', 'utg_call', 'raise']
        elif s.action_facing == 'utg_call':
            return ['check', 'raise']
        elif s.action_facing == 'bet':
            return ['fold', 'call', 'raise']
        elif s.action_facing == 'raise':
            return ['fold', 'call', 'reraise']
        elif s.action_facing == 'reraise':
            return ['fold', 'call']
        elif s.action_facing == 'check':
            return ['check', 'bet']
        return []

    """generates the intermediate state, which then the environment steps and the opponent before
        feeding back into this player"""
    @staticmethod
    def apply_action(s: MDPState, a: str):
        """
        returns new state after applying action specified. STRICTLY DETERMINISTIC

        Note that if the action results in the end of betting round or hand it returns as such

        s is current state, a is chosen action that we use to update the state

        a to_act value of -1 means the environment is the one that needs to act

        TODO: write functions lol rewrote hella code

        """
        def action_to_step(s_p: MDPState, a):
            """this is called any time move to other player to act, steps action to and pos'n forward.
            also changes action facing"""
            s_p.to_act = 1 - s_p.to_act
            # change action facing
            s_p.action_facing = a
            # KEEP hero_position FIXED

            return s_p

        # new memory
        s_p = deepcopy(s)

        # handle preflop states
        if s_p.round_stage == 'preflop':
            # TODO handle folding/terminal states
            # here we are opening
            if s.action_facing == 'none':
                # opening preflop action
                if a == 'utg_call':
                    # add 1 to pot
                    s_p.pot += 1
                    # step forward
                    s_p = action_to_step(s_p, a)
                    return s_p

                elif a == 'fold':
                    # preserves to_act so we know who folded
                    s_p.folded_player = s_p.to_act
                    s_p.to_act = -1
                    s_p.action_facing = 'fold'
                    return s_p

                elif a == 'raise':
                    s_p.pot += 2
                    # step forward
                    s_p = action_to_step(s_p, a)
                    return s_p

                else:
                    raise ValueError("Action taken when facing 'none' preflop INVALID")

            # facing utg call, we are in bb
            elif s_p.action_facing == 'utg_call':
                # option
                if a == 'check':
                    # ends betting round, flops card. action stays bb
                    s_p.to_act = -1 # means betting round over, progress to next step
                    s_p.action_facing = 'check'
                    return s_p

                elif a == 'raise':
                    s_p.pot += 2
                    # move action
                    s_p = action_to_step(s_p, a)
                    return s_p

                else:
                    raise ValueError()

            elif s_p.action_facing == 'raise':
                # could be in btn or bb
                # always reraise option
                if a == "call":
                    # end betting round
                    s_p.pot+=1
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p

                elif a == 'reraise':
                    s_p.pot+=2
                    s_p = action_to_step(s_p, a)
                    return s_p

                elif a == 'fold':
                    s_p.folded_player = s_p.to_act
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p

            elif s_p.action_facing == 'reraise':
                if a == 'call':
                    s_p.pot += 1
                    s_p.to_act = -1
                    s_p.action_facing = 'call'
                    return s_p

                elif a == 'fold':
                    s_p.folded_player = s_p.to_act
                    s_p.to_act = -1
                    s_p.action_facing = 'fold'
                    return s_p

                else:
                    raise ValueError(f"Invalid action {a} facing reraise preflop")

        # handle postflop transitions
        elif s_p.round_stage == "postflop":
            if s_p.action_facing == 'raise':
                # could be in btn or bb
                # always reraise option
                if a == "call":
                    # end betting round
                    s_p.pot+=1
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p

                elif a == 'reraise':
                    s_p.pot+=2
                    s_p = action_to_step(s_p, a)
                    return s_p

                elif a == 'fold':
                    s_p.folded_player = s_p.to_act
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p

            elif s_p.action_facing == 'none':
                # opening postflop
                # can only bet or check here
                if a == 'bet':
                    s_p.pot +=1
                    s_p.action_facing = a
                if a == 'check':
                    # nothing else happens
                    pass
                s_p = action_to_step(s_p, a)
                return s_p

            elif s_p.action_facing == 'check':
                if a == 'bet':
                    s_p.pot +=1
                    s_p = action_to_step(s_p, a)
                    return s_p
                if a == 'check':
                    # check back betting rd over
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p

            elif s_p.action_facing == 'bet':
                # call, raise, fold
                if a == 'call':
                    # betting rd over
                    s_p.pot +=1
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p
                elif a == 'raise':
                    s_p.pot += 2
                    s_p = action_to_step(s_p, a)
                    return s_p
                elif a == 'fold':
                    s_p.folded_player = s_p.to_act
                    s_p.to_act = -1
                    s_p.action_facing = 'fold'
                    return s_p

            elif s_p.action_facing == 'reraise':
                # can only call or fold
                # either way betting round over
                if a == 'call':
                    s_p.pot += 1
                    s_p.to_act = -1
                    s_p.action_facing = a
                    return s_p
                elif a == 'fold':
                    s_p.folded_player = s_p.to_act
                    s_p.action_facing = 'fold'
                    s_p.to_act = -1
                    return s_p

        # return new state
        raise ValueError(
            f"Unhandled transition: stage={s.round_stage}, "
            f"facing={s.action_facing}, to_act={s.to_act}, action={a}, "
            f"state={s.to_tuple()}"
        )


    @staticmethod
    def env_transition_dist(state: MDPState, villain_policy='uniform'):
        """
        Returns distribution over possible next states from an intermediate state.

        Each outcome is:
            (probability, next_state, reward, done)
        """

        # If hero is already to act, stop advancing.
        if state.to_act == 0:
            return [(1.0, state, 0.0, False)]

        def get_flop_dist(deck):
            """
            Returns distribution over flop RANKS, respecting card multiplicity.
            Example: {'J': 0.5, 'Q': 0.25, 'K': 0.25}
            """
            n = len(deck)
            prob_tab = {}

            for card in deck:
                rank = card[0]
                prob_tab[rank] = prob_tab.get(rank, 0.0) + 1.0 / n

            return prob_tab

        def card_rank(card):
            """
            Extract rank from a card representation like 'Jh', 'Qd', 'Kc',
            or from a public card representation like 'J', 'Q', 'K'.
            """
            if card is None:
                return None
            return str(card)[0]

        def villain_private_card(state):
            """
            In this MDP state, hero_card and villain_card are both known to the
            model because DP enumerates the full underlying state.

            This is okay for a scripted opponent inside the full transition model:
            villain's policy can condition on villain's own private card and public card.
            """
            return state.villain_card

        def villain_strength_category(state):
            """
            Coarse hand-strength bucket for villain.

            preflop:
                K = strong
                Q = medium
                J = weak

            postflop:
                pair with board = strong
                K-high = medium/strong-ish
                Q-high = medium
                J-high = weak
            """
            v_rank = card_rank(villain_private_card(state))
            board_rank = card_rank(state.flop_card)

            if state.round_stage == "preflop":
                if v_rank == "K":
                    return "strong"
                if v_rank == "Q":
                    return "medium"
                return "weak"

            # postflop
            if board_rank is not None and v_rank == board_rank:
                return "strong"

            if v_rank == "K":
                return "medium_strong"
            if v_rank == "Q":
                return "medium"
            return "weak"

        def choose_first_available(actions, preferences):
            for preferred in preferences:
                if preferred in actions:
                    return {preferred: 1.0}
            return {actions[0]: 1.0}

        def normalize(dist):
            total = sum(dist.values())
            if total <= 0:
                raise ValueError(f"Invalid zero-probability action distribution: {dist}")
            return {a: p / total for a, p in dist.items() if p > 0}

        @staticmethod
        def villain_action_dist(state, villain_policy):
            actions = LeducSimpleMDP.legal_actions_from_mdp(state)

            if villain_policy == "uniform":
                return {a: 1.0 / len(actions) for a in actions}

            if villain_policy == "always_raise":
                return choose_first_available(actions, ["raise", "bet", "call", "check"])

            if villain_policy == "always_fold":
                return choose_first_available(actions, ["fold", "check", "call"])

            if villain_policy == "check_call":
                return choose_first_available(actions, ["check", "call"])

            if villain_policy == "tight":
                return choose_first_available(actions, ["check", "fold", "call", "raise"])

            if villain_policy == "aggressive":
                return choose_first_available(actions, ["raise", "bet", "call", "check", "fold"])

            if villain_policy == "rank_aware_tight":
                strength = villain_strength_category(state)

                # Strong hands: value bet / raise.
                if strength == "strong":
                    return choose_first_available(
                        actions,
                        ["raise", "bet", "call", "check"],
                    )

                # Medium-strong hands: continue, but avoid bloating pot too much.
                if strength == "medium_strong":
                    return choose_first_available(
                        actions,
                        ["check", "call", "bet", "raise", "fold"],
                    )

                # Medium hands: check/call, fold to heavy pressure.
                if strength == "medium":
                    if state.action_facing == "reraise" and "fold" in actions:
                        return {"fold": 1.0}

                    return choose_first_available(
                        actions,
                        ["check", "call", "fold", "bet"],
                    )

                # Weak hands: check if free, fold if pressured.
                return choose_first_available(
                    actions,
                    ["check", "fold", "call"],
                )

            if villain_policy == "rank_aware_aggressive":
                strength = villain_strength_category(state)

                # Strong hands: always pressure.
                if strength == "strong":
                    return choose_first_available(
                        actions,
                        ["raise", "bet", "call", "check"],
                    )

                # Medium-strong hands: often pressure, but can continue.
                if strength == "medium_strong":
                    dist = {}

                    if "raise" in actions:
                        dist["raise"] = 0.60
                    if "bet" in actions:
                        dist["bet"] = 0.60
                    if "call" in actions:
                        dist["call"] = 0.30
                    if "check" in actions:
                        dist["check"] = 0.30
                    if "fold" in actions:
                        dist["fold"] = 0.10

                    if dist:
                        return normalize(dist)

                    return choose_first_available(actions, ["call", "check", "fold"])

                # Medium hands: semi-aggressive. Sometimes bluff/value bet, sometimes pot-control.
                if strength == "medium":
                    dist = {}

                    if "raise" in actions:
                        dist["raise"] = 0.35
                    if "bet" in actions:
                        dist["bet"] = 0.45
                    if "call" in actions:
                        dist["call"] = 0.40
                    if "check" in actions:
                        dist["check"] = 0.40
                    if "fold" in actions:
                        dist["fold"] = 0.25

                    if dist:
                        return normalize(dist)

                    return choose_first_available(actions, ["check", "call", "fold"])

                # Weak hands: bluff sometimes when opening or facing a check,
                # but mostly give up versus pressure.
                if strength == "weak":
                    dist = {}

                    if "bet" in actions:
                        dist["bet"] = 0.25
                    if "raise" in actions:
                        dist["raise"] = 0.15
                    if "check" in actions:
                        dist["check"] = 0.75
                    if "call" in actions:
                        dist["call"] = 0.20
                    if "fold" in actions:
                        dist["fold"] = 0.65

                    if dist:
                        return normalize(dist)

                    return choose_first_available(actions, ["check", "fold", "call"])

            raise ValueError(f"Unknown villain_policy: {villain_policy}")


        def fold_reward(state):
            if state.folded_player == 0:
                return -state.pot / 2.0
            else:
                return state.pot / 2.0

        def showdown_reward(state):
            winner = showdown(state.hero_card, state.villain_card, state.flop_card)

            if winner == 0:
                return state.pot / 2.0
            elif winner == 1:
                return -state.pot / 2.0
            else:
                return 0.0

        pure_deck = {"Jh", "Jd", "Qh", "Qd", "Kh", "Kd"}

        cards_out = {state.hero_card, state.villain_card}
        if state.flop_card not in [None, 'none']:
            cards_out.add(state.flop_card)

        new_deck = pure_deck - cards_out

        # --------------------------------------------------
        # CASE 1: environment / round resolver node
        # --------------------------------------------------
        if state.to_act == -1:

            # Preflop betting round ended -> enumerate flops
            if state.round_stage == "preflop":

                if state.action_facing in ["check", "call"]:
                    flop_dist = get_flop_dist(new_deck)

                    outs = []

                    for flop_rank, p in flop_dist.items():
                        state_p = deepcopy(state)

                        state_p.flop_card = flop_rank
                        state_p.round_stage = "postflop"
                        state_p.action_facing = "none"

                        # BB acts first postflop
                        state_p.to_act = 0 if state_p.hero_position == "bb" else 1

                        # If hero acts next, return directly.
                        if state_p.to_act == 0:
                            outs.append((p, state_p, 0.0, False))

                        # If villain acts next, recursively advance through villain action.
                        else:
                            sub_outs = LeducSimpleMDP.env_transition_dist(
                                state_p,
                                villain_policy=villain_policy
                            )

                            for p2, s2, r2, d2 in sub_outs:
                                outs.append((p * p2, s2, r2, d2))

                    return outs

                elif state.action_facing == "fold":
                    r = fold_reward(state)
                    return [(1.0, state, r, True)]

                else:
                    raise ValueError(f"Bad preflop env state: {state.to_tuple()}")

            # Postflop betting round ended -> showdown or fold
            elif state.round_stage == "postflop":

                if state.action_facing in ["check", "call"]:
                    r = showdown_reward(state)
                    return [(1.0, state, r, True)]

                elif state.action_facing == "fold":
                    r = fold_reward(state)
                    return [(1.0, state, r, True)]

                else:
                    raise ValueError(f"Bad postflop env state: {state.to_tuple()}")

        # --------------------------------------------------
        # CASE 2: villain to act
        # --------------------------------------------------
        if state.to_act == 1:
            dist = villain_action_dist(state, villain_policy)
            outs = []

            for villain_action, p in dist.items():
                state_mid = LeducSimpleMDP.apply_action(state, villain_action)

                sub_outs = LeducSimpleMDP.env_transition_dist(
                    state_mid,
                    villain_policy=villain_policy
                )

                for p2, s2, r2, d2 in sub_outs:
                    outs.append((p * p2, s2, r2, d2))

            return outs

        raise ValueError(f"Invalid to_act value: {state.to_act}")

    @staticmethod
    def action_outcomes(state: MDPState, action: str, villain_policy='uniform'):
        """
        Hero chooses action, then environment/villain/chance advances
        until terminal or next hero decision state.
        """
        mid_state = LeducSimpleMDP.apply_action(state, action)
        return LeducSimpleMDP.env_transition_dist(mid_state, villain_policy)




