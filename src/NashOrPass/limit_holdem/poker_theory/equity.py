def required_equity_to_call(pot_size: float, call_cost: float) -> float:
    """
    Minimum equity required for a call to break even.

    required_equity = call_cost / (pot_size + call_cost)
    """
    if pot_size < 0:
        raise ValueError("pot_size must be nonnegative")
    if call_cost <= 0:
        raise ValueError("call_cost must be positive")

    return call_cost / (pot_size + call_cost)


def call_is_profitable(equity: float, pot_size: float, call_cost: float) -> bool:
    return equity >= required_equity_to_call(pot_size, call_cost)

def get_rough_equity(private, public):
    """
    calculate the believed equity you have based solely on private and public cards
    :return:
    """
    pass