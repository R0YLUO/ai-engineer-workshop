"""The Agent: a model, a system prompt, and one call.

Everything the model knows is in `messages`: the context window. Here the
agent gets the email and answers in a single LLM call, with no tools and no
access to our systems.
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-5-5"


class Agent:
    def __init__(self, name: str, system_prompt: str, verbose: bool = True):
        self.name = name
        self.system_prompt = system_prompt
        self.verbose = verbose

        self.client = anthropic.AsyncAnthropic()
        self.messages: list[dict] = []  # the context window
        self.tool_calls: list[dict] = []  # a log of every tool call, used by the evals

    async def run(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = await self.client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=self.system_prompt,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})
        self.log_turn(1, response)
        return "".join(b.text for b in response.content if b.type == "text")

    # ---- Logging: watch the context build up ----

    def log(self, text: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {text}")

    def log_turn(self, turn: int, response) -> None:
        usage = response.usage
        self.log(
            f"── turn {turn} ── stop_reason={response.stop_reason} · "
            f"messages in context={len(self.messages)} · "
            f"input tokens={usage.input_tokens} · output tokens={usage.output_tokens}"
        )
        for block in response.content:
            if block.type == "text" and block.text.strip():
                self.log(f"  💬 {block.text.strip()}")
