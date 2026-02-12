from src.NashOrPass.environment.leduc.models.Action import Action

def generate_possible_bet_actions(prev_action: Action, bb_amnt: int, effective_bankroll: int, can_raise: bool = True):
    """
    Generate list of possible betting actions

    Args:
        prev_action: Previous action in the round (or None for opening)
        bb_amnt: Big blind amount (minimum bet/raise increment)
        effective_bankroll: Player's remaining chips
        can_raise: Whether raising is allowed (e.g., max raises not reached)

    Returns:
        List of Action objects representing valid bet actions
    """
    bets = []

    # Case 1: After a check (or opening action with no bet)
    if prev_action is None or prev_action.type == "check":
        # Can open/raise in increments of bb_amnt up to entire stack
        if effective_bankroll >= bb_amnt:
            # Generate raises: bb, 2bb, 3bb, ... up to effective_bankroll
            max_raise_multiple = effective_bankroll // bb_amnt
            bets = [
                Action("raise", amount_raised=k * bb_amnt, total_amount=k * bb_amnt)
                for k in range(1, max_raise_multiple + 1)
            ]
        else:
            # Only option is all-in for less than bb
            bets = [Action("raise", amount_raised=effective_bankroll, total_amount=effective_bankroll)]

    # Case 2: After a raise (need to call or re-raise)
    elif prev_action.type == "raise":
        prev_bet_sz = prev_action.total_amount
        prev_raise_amnt = prev_action.amount_raised

        # Can we even call?
        if effective_bankroll >= prev_bet_sz:
            # Option 1: Call the bet
            bets.append(Action("call", amount_raised=0, total_amount=prev_bet_sz))

            # Option 2: Re-raise (if allowed and we have chips left)
            if can_raise:
                chips_after_call = effective_bankroll - prev_bet_sz

                if chips_after_call >= prev_raise_amnt:
                    # Minimum re-raise is prev_raise_amnt
                    # Generate re-raises: min_raise, 2x min_raise, etc.
                    max_raise_multiple = chips_after_call // prev_raise_amnt

                    for k in range(1, max_raise_multiple + 1):
                        raise_amnt = k * prev_raise_amnt
                        total_amnt = prev_bet_sz + raise_amnt
                        bets.append(Action("raise", amount_raised=raise_amnt, total_amount=total_amnt))

                # Edge case: can call but not min-raise, can still go all-in
                elif chips_after_call > 0:
                    # All-in raise for less than minimum
                    bets.append(Action("raise", amount_raised=chips_after_call, total_amount=effective_bankroll))

        else:
            # Can't even call - only option is all-in for less (if any chips)
            if effective_bankroll > 0:
                bets.append(Action("call", amount_raised=0, total_amount=effective_bankroll))

    return bets