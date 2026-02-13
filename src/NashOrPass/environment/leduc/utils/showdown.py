def showdown(card1, card2, flop):
    """
    Determine the winner at showdown in Leduc Hold'em

    Args:
        card1: Player 1's card (e.g., "Jh")
        card2: Player 2's card (e.g., "Qd")
        flop: Community card (e.g., "Kh")

    Returns:
        winning player index (given that args passed in corresponding order), 2 if chop
    """
    # Check if either player paired with the flop
    if card1[0] == flop[0]:
        return 0
    elif card2[0] == flop[0]:
        return 1

    # neither paired, check rank
    rank_order = {'J': 0, 'Q': 1, 'K': 2}

    rank1 = rank_order[card1[0]]
    rank2 = rank_order[card2[0]]

    if rank1 > rank2:
        return 0
    elif rank2 > rank1:
        return 1
    else:

        return 2