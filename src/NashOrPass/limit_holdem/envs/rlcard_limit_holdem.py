import rlcard


ACTION_ID_TO_NAME = {
    0: "call",
    1: "raise",
    2: "fold",
    3: "check",
}

ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}


def make_env(seed: int | None = None):
    """
    Create an RLCard Limit Texas Hold'em environment.
    """
    config = {}
    if seed is not None:
        config["seed"] = seed
    return rlcard.make("limit-holdem", config=config)


def get_legal_actions(state) -> list[int]:
    """
    RLCard states contain a legal_actions dictionary.
    Return the legal action ids.
    """
    return list(state["legal_actions"].keys())


def action_name(action_id: int) -> str:
    return ACTION_ID_TO_NAME.get(action_id, f"unknown_{action_id}")


def action_id(name: str) -> int:
    return ACTION_NAME_TO_ID[name]


def get_obs(state):
    """
    Return the vector observation from an RLCard state.
    """
    return state["obs"]