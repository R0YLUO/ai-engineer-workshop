"""The email workflow: a DETERMINISTIC router in front of NON-DETERMINISTIC agents.

    email ──► classify (1 LLM call, fixed output schema)
                 │
                 ├── "dispute" ──► Dispute Agent  (agent loop + MCP tools: db lookups, chat)
                 │
                 └── "enquiry" ──► RAG retrieval (plain code) ──► Enquiry Agent (MCP tool: chat only)

Deterministic = the path is decided by OUR code (if/else). Same input, same steps.
Non-deterministic = the agent decides which tools to call, in what order, how many times.

Use code for the parts you can specify, and let the model handle the parts you can't.
"""

from typing import Literal

import anthropic
from pydantic import BaseModel

from harness.agent import MODEL, Agent
from rag.embedding import build_vector_store
from rag.retrieval import search

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


# ---- The enquiry agent: answers from retrieved documentation ----

ENQUIRY_AGENT_PROMPT = """You are a customer service assistant at Swift Freight Co., a delivery services company.
You answer general enquiries about our services using ONLY the documentation excerpts provided.

Post ONE message to the customer service team chat using write_message. It must contain:
- SUMMARY: one line on what the customer is asking
- ANSWER NOTES: the relevant facts from the documentation, citing the section headings
- GAPS: anything asked that the documentation does not cover (or "None")
- DRAFT REPLY: a reply email to the customer, polite and formal, signed "Swift Freight Customer Service"

Never invent policies, prices or capabilities that are not in the documentation.
A human reviews your message before anything is sent to the customer."""

_vector_store = None


def retrieve_context(question: str, k: int = 4) -> str:
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store()
    results = search(_vector_store, question, k=k)
    return "\n\n".join(chunk for _score, chunk in results)


# ---- The router: a single, structured LLM call (not an agent) ----


class EmailClassification(BaseModel):
    category: Literal["dispute", "enquiry"]
    reason: str


CLASSIFIER_PROMPT = """You triage emails sent to Swift Freight Co., a delivery services company.
Classify the email as exactly one of:
- "dispute": the customer is questioning or complaining about an invoice, charge or amount billed.
- "enquiry": a general question about our services, policies, pricing model or capabilities."""


async def classify_email(email: str) -> EmailClassification:
    response = await anthropic.AsyncAnthropic().messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": email}],
        output_format=EmailClassification,  # the response MUST match this schema
    )
    return response.parsed_output


# ---- The workflow ----


async def handle_email(email: str, verbose: bool = True) -> Agent:
    """Handle one incoming email and return the agent that handled it."""
    classification = await classify_email(email)
    if verbose:
        print(f"[Router] category={classification.category} · reason={classification.reason}\n")

    if classification.category == "dispute":
        agent = Agent("Dispute Agent", DISPUTE_AGENT_PROMPT, verbose=verbose)
        task = f"A new email has arrived. Investigate it.\n\n<email>\n{email}\n</email>"
    else:
        context = retrieve_context(email)
        agent = Agent("Enquiry Agent", ENQUIRY_AGENT_PROMPT, allowed_tools=["write_message"], verbose=verbose)
        task = (
            f"<documentation>\n{context}\n</documentation>\n\n"
            f"A new email has arrived. Answer it.\n\n<email>\n{email}\n</email>"
        )

    async with agent:
        await agent.run(task)
    return agent
