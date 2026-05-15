# src/NashOrPass/leduc/agents/llm_agent.py

import os
import json
import random
from collections import Counter
from dotenv import load_dotenv

from NashOrPass.leduc.agents.models.MDP import LeducSimpleMDP

load_dotenv()

class LeducLLMAgent:
    def __init__(
        self,
        model_name,
        player_name="LLM",
        temperature=0.2,
        hist_capacity=5,
        fallback_policy="check_call",
        verbose=False,
    ):
        self.model_name = model_name
        self.player_name = player_name
        self.temperature = temperature
        self.hist_capacity = hist_capacity
        self.fallback_policy = fallback_policy
        self.verbose = verbose

        self.history = []
        self.stats = Counter()
        self.last_info = None

    def reset_history(self):
        self.history = []
        self.last_info = None

    def choose_action(self, state):
        legal_actions = LeducSimpleMDP.legal_actions_from_mdp(state)

        prompt = self.build_prompt(
            state=state,
            legal_actions=legal_actions,
            history=self.history[-self.hist_capacity:],
        )

        try:
            result = self.invoke_structured(prompt, legal_actions)
            action = result.get("action")
            reason = result.get("reason", "")
            bluff = bool(result.get("bluff", False))
        except Exception as e:
            self.stats["api_or_parse_errors"] += 1
            if self.verbose:
                print(f"[{self.model_name}] error:", repr(e))

            action = None
            reason = "API or parse error."
            bluff = False

        if action not in legal_actions:
            self.stats["invalid_actions"] += 1
            action = self.fallback_action(legal_actions)

        self.stats[f"action:{action}"] += 1
        if bluff:
            self.stats["self_reported_bluffs"] += 1

        self.last_info = {
            "model": self.model_name,
            "action": action,
            "legal_actions": legal_actions,
            "reason": reason,
            "bluff": bluff,
            "state": state.to_tuple(),
        }

        self.history.append(self.last_info)

        if self.verbose:
            print(f"[{self.model_name}] {action}; bluff={bluff}; {reason}")

        return action

    def invoke_structured(self, prompt, legal_actions):
        raise NotImplementedError

    def fallback_action(self, legal_actions):
        if self.fallback_policy == "random":
            return random.choice(legal_actions)

        for preferred in ["check", "call", "fold", "bet", "raise", "reraise", "utg_call"]:
            if preferred in legal_actions:
                return preferred

        return legal_actions[0]

    @staticmethod
    def build_prompt(state, legal_actions, history):
        return f"""
You are playing heads-up Leduc Hold'em.

Rules summary:
- Deck has two Jacks, two Queens, and two Kings.
- Each player has one private card.
- One public flop card may appear postflop.
- Pair beats high card. Otherwise K > Q > J.
- This is limit betting, so choose only from the legal actions.

Your current private card:
{state.hero_card}

Known public flop card:
{state.flop_card}

Current state:
- round_stage: {state.round_stage}
- action_facing: {state.action_facing}
- hero_position: {state.hero_position}

- pot: {state.pot}

Legal actions:
{legal_actions}

Recent history:
{json.dumps(history, indent=2, default=str)}

Return ONLY JSON:
{{
  "action": one of {legal_actions},
  "reason": "one short poker explanation",
  "bluff": true or false
}}

Set bluff=true only if you bet/raise mainly because you expect folds despite likely not having the best hand.
"""


def parse_json_object(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found: {text}")

    return json.loads(text[start:end + 1])


class GPTLeducAgent(LeducLLMAgent):
    def __init__(
        self,
        model_name="gpt-4.1-mini",
        temperature=0.2,
        hist_capacity=5,
        fallback_policy="check_call",
        verbose=False,
    ):
        super().__init__(
            model_name=model_name,
            player_name="GPT",
            temperature=temperature,
            hist_capacity=hist_capacity,
            fallback_policy=fallback_policy,
            verbose=verbose,
        )

        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def invoke_structured(self, prompt, legal_actions):
        schema = {
            "name": "leduc_action",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": legal_actions},
                    "reason": {"type": "string"},
                    "bluff": {"type": "boolean"},
                },
                "required": ["action", "reason", "bluff"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Leduc Hold'em poker agent. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
        )

        return parse_json_object(resp.choices[0].message.content)


class ClaudeLeducAgent(LeducLLMAgent):
    def __init__(
        self,
        model_name="claude-sonnet-4-6",
        temperature=0.2,
        hist_capacity=5,
        fallback_policy="check_call",
        verbose=False,
    ):
        super().__init__(
            model_name=model_name,
            player_name="Claude",
            temperature=temperature,
            hist_capacity=hist_capacity,
            fallback_policy=fallback_policy,
            verbose=verbose,
        )

        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def invoke_structured(self, prompt, legal_actions):
        """
        Claude version.

        We force Claude to call a fake tool called `choose_leduc_action`.
        The tool input is the structured object we want:
            {
                "action": one of legal_actions,
                "reason": string,
                "bluff": boolean
            }
        """
        tool = {
            "name": "choose_leduc_action",
            "description": "Choose exactly one legal Leduc Hold'em action.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": legal_actions,
                        "description": "The chosen legal action.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief poker reasoning for the action.",
                    },
                    "bluff": {
                        "type": "boolean",
                        "description": "Whether this action is a bluff or semi-bluff.",
                    },
                },
                "required": ["action", "reason", "bluff"],
                "additionalProperties": False,
            },
        }

        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=512,
            temperature=self.temperature,
            system=(
                "You are a Leduc Hold'em poker agent. "
                "You must choose exactly one legal action by calling the tool."
            ),
            messages=[
                {"role": "user", "content": prompt},
            ],
            tools=[tool],
            tool_choice={
                "type": "tool",
                "name": "choose_leduc_action",
            },
        )

        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)

        raise ValueError(f"Claude did not return a tool call. Raw response: {resp}")


class GrokLeducAgent(LeducLLMAgent):
    def __init__(
        self,
        model_name="grok-4.3",
        temperature=0.2,
        hist_capacity=5,
        fallback_policy="check_call",
        verbose=False,
    ):
        super().__init__(
            model_name=model_name,
            player_name="Grok",
            temperature=temperature,
            hist_capacity=hist_capacity,
            fallback_policy=fallback_policy,
            verbose=verbose,
        )

        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
        )

    def invoke_structured(self, prompt, legal_actions):
        """
        Grok version using xAI's OpenAI-compatible API.

        This mirrors the GPT agent style as closely as possible.
        """
        schema = {
            "name": "leduc_action",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": legal_actions},
                    "reason": {"type": "string"},
                    "bluff": {"type": "boolean"},
                },
                "required": ["action", "reason", "bluff"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Leduc Hold'em poker agent. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
        )

        return parse_json_object(resp.choices[0].message.content)