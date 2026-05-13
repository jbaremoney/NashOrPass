
from NashOrPass.limit_holdem.agents.random import RandomAgent
from NashOrPass.limit_holdem.agents.check_call import CheckCallAgent
from NashOrPass.limit_holdem.agents.agg import AggressiveAgent
from NashOrPass.limit_holdem.agents.tight import TightAgent


def make_agent(name: str):
    name = name.lower()

    if name == "random":
        return RandomAgent()
    if name in {"check_call", "check-call", "checkcall"}:
        return CheckCallAgent()
    if name == "aggressive":
        return AggressiveAgent()
    if name == "tight":
        return TightAgent()


    raise ValueError(f"Unknown agent name: {name}")