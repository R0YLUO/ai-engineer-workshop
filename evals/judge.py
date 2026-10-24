"""LLM-as-judge: use a model to grade things code can't check (tone, politeness).

Keep judges narrow: one question, a clear rubric, a structured verdict.
"""

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from harness.agent import MODEL

load_dotenv()


class Verdict(BaseModel):
    passed: bool
    reason: str


TONE_RUBRIC = """You grade draft customer emails written by a delivery company's finance and customer service teams.
Find the DRAFT REPLY in the message below and judge ONLY that draft.

It passes if ALL of these are true:
- It is polite: courteous, acknowledges the customer, no blame or sarcasm.
- It is formal: professional business register, no slang, no emojis.
- It has a greeting and a sign-off.

It fails if any one of these is false. Give a one-sentence reason."""


def judge_tone(message: str) -> Verdict:
    response = anthropic.Anthropic().messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=TONE_RUBRIC,
        messages=[{"role": "user", "content": f"<message>\n{message}\n</message>"}],
        output_format=Verdict,
    )
    return response.parsed_output
