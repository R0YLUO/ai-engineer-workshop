"""The email workflow: what happens when an email arrives.

harness/main.py reads an email file and calls handle_email().
"""

from harness.agent import Agent

# ---- The dispute agent: investigates invoice disputes using our systems ----

DISPUTE_AGENT_PROMPT = """You are a finance operations assistant at Swift Freight Co., a delivery services company.
You investigate customer emails that dispute an invoice.

Your job:
1. Use the tools to find the customer, their contracted pricing, and the booking and invoice in question.
2. Recalculate what the invoice SHOULD have been, using the customer's contract:
   line total = distance_km × contracted rate per km for the service level booked + applicable surcharges
   invoice total = line total × (1 − contracted discount % / 100)
3. Compare with what was actually charged and pinpoint exactly which value was applied incorrectly.
4. Post ONE message to the finance team chat using write_message. It must contain:
   - SUMMARY: one line on what the customer is disputing
   - FINDINGS: the values applied vs the contracted values, with the calculation and the correct total
   - RECOMMENDED ACTION: what the finance team should do
   - DRAFT REPLY: a reply email to the customer, polite and formal, signed "Swift Freight Finance Team"

You never contact the customer directly; a human reviews your message first.
If the data does not support the customer's claim, say so clearly in your findings."""


async def handle_email(email: str, verbose: bool = True) -> Agent:
    """Handle one incoming email and return the agent that handled it."""
    agent = Agent("Dispute Agent", DISPUTE_AGENT_PROMPT, verbose=verbose)
    async with agent:  # connects to the MCP server and loads its tools
        await agent.run(f"A new email has arrived. Investigate it.\n\n<email>\n{email}\n</email>")
    return agent
