"""Lesson 1: one call to Claude.

An LLM API is stateless: you send a list of messages, you get one message back.
There is no memory between calls - if you want a conversation, YOU resend the
history every time.

    uv run python harness/examples/01_first_call.py
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

messages = [{"role": "user", "content": "In one sentence, what does a freight invoice contain?"}]
response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    system="You are a concise assistant for a delivery services company.",
    messages=messages,
)
answer = response.content[-1].text
print("Claude:", answer)
print("Usage:", response.usage.input_tokens, "input tokens,", response.usage.output_tokens, "output tokens")

# A follow-up question only works because we send the whole history again.
messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": "Which of those parts most often causes billing disputes?"})
response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    system="You are a concise assistant for a delivery services company.",
    messages=messages,
)
print("\nClaude:", response.content[-1].text)
print("Usage:", response.usage.input_tokens, "input tokens  ← grew, because the history was resent")
