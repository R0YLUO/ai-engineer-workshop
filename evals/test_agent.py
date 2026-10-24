"""Evals for the email workflow.

Run:  uv run pytest evals -v
Each email is run through the workflow ONCE and shared by all tests (agents are slow and cost money).
"""

import asyncio
from pathlib import Path

import pytest

from evals.judge import judge_tone
from harness.workflow import handle_email

EMAILS = Path(__file__).parent.parent / "data" / "emails"

CASES = {
    "dispute_acme.txt": {
        "agent": "Dispute Agent",
        "tools": {"lookup_customer", "get_booking_details", "write_message"},
        # Contract says $2.50/km standard: 120 km × 2.50 = $300, not $480 (express rate was applied).
        "must_mention": ["INV-1042", "300", "480"],
    },
    "dispute_coastal.txt": {
        "agent": "Dispute Agent",
        "tools": {"lookup_customer", "get_booking_details", "write_message"},
        # (80 km × 3.10 + 40 heavy item) × 0.9 = $259.20, the 10% discount was missing.
        "must_mention": ["INV-1051", "259.20", "10%"],
    },
    "enquiry_refrigerated.txt": {
        "agent": "Enquiry Agent",
        "tools": {"write_message"},
        "must_mention": ["8°C", "log"],
    },
    "enquiry_cancellation.txt": {
        "agent": "Enquiry Agent",
        "tools": {"write_message"},
        "must_mention": ["4 hours", "$35", "aerosol"],
    },
}


@pytest.fixture(scope="module")
def agents():
    """Run every email through the workflow once."""
    return {
        filename: asyncio.run(handle_email((EMAILS / filename).read_text(), verbose=False))
        for filename in CASES
    }


def posted_message(agent) -> str:
    messages = [call["input"]["message"] for call in agent.tool_calls if call["name"] == "write_message"]
    assert messages, "the agent did not post a message to the chat"
    return messages[0]


@pytest.mark.parametrize("email", CASES)
def test_routed_to_the_right_agent(agents, email):
    assert agents[email].name == CASES[email]["agent"]


@pytest.mark.parametrize("email", CASES)
def test_agent_wrote_one_message(agents, email):
    posted = [call for call in agents[email].tool_calls if call["name"] == "write_message"]
    assert len(posted) == 1


@pytest.mark.parametrize("email", CASES)
def test_agent_used_the_right_tools(agents, email):
    tools_used = {call["name"] for call in agents[email].tool_calls}
    assert tools_used == CASES[email]["tools"]


@pytest.mark.parametrize("email", CASES)
def test_message_holds_the_right_data(agents, email):
    message = posted_message(agents[email])
    for expected in CASES[email]["must_mention"]:
        assert expected.lower() in message.lower(), f"missing {expected!r} in:\n{message}"


@pytest.mark.parametrize("email", CASES)
def test_draft_reply_is_polite_and_formal(agents, email):
    verdict = judge_tone(posted_message(agents[email]))
    assert verdict.passed, verdict.reason
