def get_hand_equity(dealt, flop=None):
    if flop is None:
        if (dealt == "Jh" or dealt == "Jd"):
            return .3
        elif (dealt == "Qh" or dealt == "Qd"):
            return .5
        else:
            return .7

    if (flop == "Jh" or flop == "Jd"):
        if (dealt == "Jh" or dealt == "Jd"):
            return 1
        elif (dealt == "Qh" or dealt == "Qd"):
            return .125
        else:
            return .625

    elif (flop == "Qh" or flop == "Qd"):
        if (dealt == "Jh" or dealt == "Jd"):
            return .125
        elif (dealt == "Qh" or dealt == "Qd"):
            return 1
        else:
            return .625

    else:  # king is flopped
        if (dealt == "Jh" or dealt == "Jd"):
            return .125
        elif (dealt == "Kh" or dealt == "Kd"):
            return 1
        else:
            return .625