"""
code to use LLM to play poker

can vary how many previous hands it has seen all in context window

can also vary prompt you use to customize strategy
"""


def get_prompt_from_state(state):
    """
    only build the prompt for most current information, basically private cards public cards pot size, and most recent
    oppt action

    we can then vary how much of the history we pass on each model call, like how much is in context window

    ie perhaps we don't pass any stuff we don't need to from current hand, or perhaps we pass whole hand history
    or perhaps we maintain the past three hands, or 10, etc.

    Then we can compare performance

    """


def build_legal_actions(state):
    """
    should return a schema and we use the schema to generate structured output with model call
    ... we use this to get the action selected from model
    """


def get_model_action(prompt, legal_actions, history=None):
    """
    feeds prompt and perhaps history to llm, and forces a selection of one of the legal actions
    """