"""Lesson 2: tool use, and the loop that turns a model into an agent.

Claude can't run code. It can only ASK us to run a tool by replying with a
`tool_use` block (stop_reason == "tool_use"). We run the tool, send back a
`tool_result`, and call Claude again. Repeat until it stops asking.

    uv run python harness/examples/02_tool_use.py
"""

import json

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# 1. A plain Python function...
INVOICES = {"INV-1042": {"customer": "Acme Retail", "distance_km": 120, "rate_per_km": 4.00, "total": 480.00}}


def get_invoice(invoice_id: str) -> dict:
    return INVOICES.get(invoice_id, {"error": f"No invoice {invoice_id}"})


# 2. ...described to Claude with a name, a description and a JSON schema for its inputs.
tools = [
    {
        "name": "get_invoice",
        "description": "Look up an invoice by its id (e.g. INV-1042). Returns distance, rate applied and total.",
        "input_schema": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
        },
    }
]

# 3. The agentic loop.
messages = [{"role": "user", "content": "Was INV-1042 charged at Acme's standard rate of $2.50/km?"}]
while True:
    response = client.messages.create(
        model="claude-haiku-5-5", max_tokens=4096, tools=tools, messages=messages
    )
    messages.append({"role": "assistant", "content": response.content})
    print(f"stop_reason={response.stop_reason} · messages in context={len(messages)}")

    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = get_invoice(**block.input)
            print(f"  🔧 Claude called {block.name}({block.input}) → {result}")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )
    messages.append({"role": "user", "content": tool_results})

print("\nClaude:", response.content[-1].text)
