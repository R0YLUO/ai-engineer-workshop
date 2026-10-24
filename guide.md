# Workshop Guide

You'll build an AI agent that triages customer emails for Swift Freight's finance team. You start with a stub and build one piece at a time. After every step you **run it and look at the output**, because seeing what each piece adds (and what is still missing) is the point.

| Step | You build | What you should see |
|---|---|---|
| 0 | Nothing yet: explore the starting state | `NotImplementedError` |
| 1 | An agent: a system prompt and one LLM call | A polite but useless answer, because the agent has no data |
| 2 | Evals | **0 / 16** passing |
| 3 | Nothing: explore the MCP server by calling it directly | The tools work, and a message appears in the chat app |
| 4 | Give the agent tools (still one LLM call) | Claude *asks* for a tool, then stops |
| 5 | The agentic loop | Disputes resolved and posted to chat. Evals: **12 / 16** |
| 6 | RAG: chunk → embed → vector distance → retrieve | The right documentation sections for a question |
| 7 | Classifier and routing to an enquiry agent | Evals: **20 / 20** |

Background for each concept is in [`docs/lecture.md`](docs/lecture.md).

## How checkpoints work

`main` is the starting state. Its history holds one commit per step, building up to the finished solution, and each of those commits has a tag with the finished code for that step: `step-0` … `step-5`, `step-6.1` … `step-6.4`, `step-7`.

```bash
git checkout -b workshop main                         # start here
git stash -u && git checkout -B workshop step-4       # stuck? jump to the end of any step (step-4 here)
git diff step-4 step-5                                # see exactly what a step changes
git checkout step-7                                   # look at the finished solution
```

## Setup

You need [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
uv sync
cp .env.example .env        # paste your ANTHROPIC_API_KEY into .env
```

The model is set once, in `MODEL` in `harness/agent.py`.

---

## Step 0: The starting state

```bash
uv run python -m harness.main data/emails/dispute_acme.txt
```

You get `NotImplementedError: Step 1: build the agent`. Before building anything, look around:

- `harness/main.py` reads an email file and calls `handle_email()` in `harness/workflow.py`. That function is the stub you'll build out.
- `data/emails/` holds four emails: two invoice **disputes** and two general **enquiries**.
- `mcp_server/` holds Swift Freight's internal systems: a customer/booking/invoice database (`db.py`) and a team chat. **This part is already built for you.**
- `data/documents/service_guide.md` is the long document customers ask questions about.

Read `dispute_acme.txt`, then find Acme's booking in `mcp_server/db.py`. Can you see what went wrong with the invoice? That's what the agent has to work out.

---

## Step 1: An agent with a system prompt and one LLM call

**Concept:** the LLM API is stateless. You send a system prompt and a list of messages, and you get one message back. Everything the model knows is in that request: the **context window**.

Warm-up (optional): `uv run python harness/examples/01_first_call.py`

**Create `harness/agent.py`:**

```python
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
```

**Replace the stub in `harness/workflow.py` with:**

```python
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
    await agent.run(f"A new email has arrived. Investigate it.\n\n<email>\n{email}\n</email>")
    return agent
```

**Run it:**

```bash
uv run python -m harness.main data/emails/dispute_acme.txt
```

**What you should see:** one turn, `stop_reason=end_turn`. Claude says it can't check the contract or the invoice, and produces at best a generic holding reply. It's polite but useless, because the agent has **no data**. The prompt tells it to use tools and post to chat, and it can do neither. A good prompt can't make up for missing information.

✅ Checkpoint: `step-1`

---

## Step 2: Evals

**Concept:** before improving the agent, decide how you'll **measure** it. An eval runs the real agent on known emails and checks properties of the result, from strict to fuzzy:

1. Did it post a message? (a side effect happened)
2. Did it use the right tools? (the trajectory)
3. Does the message contain the right numbers? (content)
4. Is the draft reply polite and formal? (quality, graded by an **LLM-as-judge**)

**Create `evals/__init__.py`** (empty).

**Create `evals/judge.py`:**

```python
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
```

**Create `evals/test_agent.py`:**

```python
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
        "tools": {"lookup_customer", "get_booking_details", "write_message"},
        # Contract says $2.50/km standard: 120 km × 2.50 = $300, not $480 (express rate was applied).
        "must_mention": ["INV-1042", "300", "480"],
    },
    "dispute_coastal.txt": {
        "tools": {"lookup_customer", "get_booking_details", "write_message"},
        # (80 km × 3.10 + 40 heavy item) × 0.9 = $259.20, the 10% discount was missing.
        "must_mention": ["INV-1051", "259.20", "10%"],
    },
    "enquiry_refrigerated.txt": {
        "tools": {"write_message"},
        "must_mention": ["8°C", "log"],
    },
    "enquiry_cancellation.txt": {
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
```

**Run it:**

```bash
uv run pytest evals -v
```

**What you should see:** **16 failed**. The agent never posts a message or calls a tool, so nothing can pass. This is your baseline, and every step from here should move this number.

**Discuss:** the enquiry emails are going to a *dispute* agent. Which of these tests do you expect to stay red until we fix that?

✅ Checkpoint: `step-2`

---

## Step 3: Explore the MCP server

**Concept:** MCP (Model Context Protocol) is a standard way to expose tools. The server says "here are my tools" and "run this tool for me". Any MCP client can use it: our agent, Claude Code, or a command-line tool. **You don't build the server. You test it by calling it directly.**

Open `mcp_server/server.py`. Each tool is a plain Python function. Its **docstring and type hints are the tool definition** the model will read.

From here on, keep these two running in their own terminals:

```bash
# Terminal 1: the MCP server, a separate process on port 8001
uv run python mcp_server/server.py

# Terminal 2: the team chat app; open http://localhost:8000
uv run python -m http.server 8000 -d mcp_server/chat_app
```

In a third terminal, call the server directly:

```bash
# What tools does it offer? (This is exactly what our agent will receive.)
uv run fastmcp list http://127.0.0.1:8001/mcp

# Call them by hand, like the agent will
uv run fastmcp call http://127.0.0.1:8001/mcp lookup_customer name=acme
uv run fastmcp call http://127.0.0.1:8001/mcp get_booking_details customer_id=C001
uv run fastmcp call http://127.0.0.1:8001/mcp write_message message="Hello from the CLI"
```

**What you should see:** the customer's contracted rates, their bookings with the values actually invoiced, and your message appearing in the chat app within 2 seconds.

**Try it:** do the agent's job yourself. Using only these calls, work out what Acme should have been charged for INV-1042.

To clear the chat: `rm mcp_server/chat_app/messages.json`

✅ Checkpoint: `step-3` (no code changes)

---

## Step 4: Give the agent tools (still one LLM call)

**Concept:** a tool is a name, a description and an input schema. Claude never runs code. It **asks** you to run a tool by replying with a `tool_use` block and `stop_reason="tool_use"`. Our agent gets its tools from the MCP server: **connect → list tools → convert them to Claude's format**.

Make sure the MCP server is running (terminal 1).

**Edit `harness/agent.py`:**

1. Replace the top of the file (docstring, imports and constants) with:

```python
"""The Agent: a model, a system prompt, and tools from an MCP server.

On start we connect to the MCP server, ask which tools it has, and hand those
definitions to Claude. Claude can now ASK for a tool (a `tool_use` block); we
run it on the MCP server. Start the server first:  uv run python mcp_server/server.py
"""

import json

import anthropic
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

MODEL = "claude-haiku-5-5"
MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"
```

2. In `__init__`, after `self.client = ...`, add:

```python
        self.mcp = Client(MCP_SERVER_URL)
        self.tools: list[dict] = []
```

3. Add the MCP methods above `run`:

```python
    # ---- MCP setup: connect, discover tools, convert them for Claude ----

    async def __aenter__(self):
        await self.mcp.__aenter__()
        mcp_tools = await self.mcp.list_tools()
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in mcp_tools
        ]
        self.log(f"Connected to MCP server. Tools: {[t['name'] for t in self.tools]}")
        return self

    async def __aexit__(self, *exc):
        await self.mcp.__aexit__(*exc)

    async def call_tool(self, name: str, tool_input: dict) -> tuple[str, bool]:
        result = await self.mcp.call_tool(name, tool_input, raise_on_error=False)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return text, result.is_error
```

4. Replace `run` (and the section comment above it) with a version that passes the tools and runs whatever Claude asks for:

```python
    # ---- One call to Claude, with tools ----

    async def run(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = await self.client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=self.system_prompt,
            tools=self.tools,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})
        self.log_turn(1, response)

        # Run every tool Claude asked for. Claude never sees the results.
        for block in response.content:
            if block.type == "tool_use":
                output, is_error = await self.call_tool(block.name, block.input)
                self.tool_calls.append({"name": block.name, "input": block.input, "output": output})
                self.log(f"  🔧 {block.name}({json.dumps(block.input)})\n     → {output[:300]}")

        return "".join(b.text for b in response.content if b.type == "text")
```

**Edit `handle_email` in `harness/workflow.py`** so the agent connects to MCP before running:

```python
    agent = Agent("Dispute Agent", DISPUTE_AGENT_PROMPT, verbose=verbose)
    async with agent:  # connects to the MCP server and loads its tools
        await agent.run(f"A new email has arrived. Investigate it.\n\n<email>\n{email}\n</email>")
    return agent
```

**Run it:**

```bash
uv run python -m harness.main data/emails/dispute_acme.txt
```

**What you should see:**

```
[Dispute Agent] Connected to MCP server. Tools: ['lookup_customer', 'get_booking_details', 'write_message']
[Dispute Agent] ── turn 1 ── stop_reason=tool_use · messages in context=2 · ...
[Dispute Agent]   🔧 lookup_customer({"name": "Acme Retail"})
     → [{"customer_id":"C001","name":"Acme Retail", ... }]
```

Then it **stops**. Claude chose a sensible first tool and MCP returned the right data, but **Claude never saw the result**, so it can't take the next step. Run the evals: still **16 failed**.

**Discuss:** what would you need to do with that tool result to let Claude continue?

✅ Checkpoint: `step-4`

---

## Step 5: The agentic loop

**Concept:** send the tool results back and call Claude again, and keep going until it stops asking for tools. **An agent is a model using tools in a loop.** Each turn appends to `messages`, so the context grows.

Warm-up (optional): `uv run python harness/examples/02_tool_use.py` shows the same loop in 60 lines.

**Update the module docstring at the top of `harness/agent.py`:**

```python
"""The Agent: a model, a system prompt, some tools, and a loop.

    ┌──────────────────────────────────────────────────────────────┐
    │  messages = [user message]                                   │
    │  loop:                                                       │
    │     response = claude(system, tools, messages)               │
    │     messages += assistant response                           │
    │     if no tool calls: done                                   │
    │     run each tool (via MCP) → messages += tool results       │
    └──────────────────────────────────────────────────────────────┘

That's the whole "agent". Everything the model knows on each turn is in
`messages` - watch it grow by running with verbose=True.

Tools come from an MCP server over HTTP: on start we connect, ask the server
which tools it has, and hand those definitions to Claude. When Claude asks to
use a tool, we forward the call to the MCP server and return the result.
Start the server first:  uv run python mcp_server/server.py
"""
```

**Replace `run` (and the section comment above it) in `harness/agent.py`:**

```python
    # ---- The agentic loop ----

    async def run(self, user_message: str, max_turns: int = 10) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for turn in range(1, max_turns + 1):
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.messages,
            )
            # The assistant's reply (text + tool calls) becomes part of the context.
            self.messages.append({"role": "assistant", "content": response.content})
            self.log_turn(turn, response)

            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            # Run every tool Claude asked for, and send ALL results back in ONE message.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output, is_error = await self.call_tool(block.name, block.input)
                self.tool_calls.append({"name": block.name, "input": block.input, "output": output})
                self.log(f"  🔧 {block.name}({json.dumps(block.input)})\n     → {output[:300]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
            self.messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"{self.name} did not finish within {max_turns} turns")
```

**Run it** with the chat app open:

```bash
uv run python -m harness.main data/emails/dispute_acme.txt
uv run python -m harness.main data/emails/dispute_coastal.txt
```

**What you should see:** about 4 turns: `lookup_customer` → `get_booking_details` → `write_message` → `end_turn`. Watch **messages in context** and **input tokens** grow every turn. A message with SUMMARY / FINDINGS / RECOMMENDED ACTION / DRAFT REPLY appears in the chat:

- Acme: express rate applied to a standard booking. **$300.00**, not $480.00.
- Coastal: contracted 10% discount missing. **$259.20**, not $288.00.

Nobody told the agent which tools to call or in what order. It decided for itself. This is the **non-deterministic** part.

**Run the evals:** `uv run pytest evals -v`. You should get **12 passed, 4 failed**. Both disputes pass everything. The enquiries fail on *tools* (the dispute agent goes looking in the database) and *data* (it has no documentation to answer from). The next two steps fix that.

The evals run the real agent, so their messages appear in the chat too. Clear it with `rm mcp_server/chat_app/messages.json`.

**Try it:**
- After the run, print `agent.messages` in `harness/main.py`. That list is everything the model saw.
- Set `max_turns=2`. What happens, and why does a real harness need this limit?

✅ Checkpoint: `step-5`

---

## Step 6: RAG

**Concept:** the enquiry emails need facts from `data/documents/service_guide.md`. The document is too long and noisy to paste into every prompt, so we put only the **relevant pieces** in front of the model. RAG has three parts: **chunk** the document, **embed** the chunks as vectors, then **retrieve** the chunks closest in meaning to the question.

The files in `rag/` are already scaffolded. Each has a `__main__` block so you can check every step. Fill in the functions marked `NotImplementedError`.

### 6.1 Chunking: `rag/chunking.py`

Split on `## ` headings, split long sections on paragraphs, and prefix each chunk with its heading so it keeps its meaning.

```python
def chunk_document(text: str, max_chars: int = 800) -> list[str]:
    chunks = []
    for section in text.split("\n## ")[1:]:  # [0] is the document title
        heading, _, body = section.partition("\n")
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) > max_chars:
                chunks.append(f"## {heading}\n{current.strip()}")
                current = ""
            current += paragraph + "\n\n"
        if current:
            chunks.append(f"## {heading}\n{current.strip()}")
    return chunks
```

Check: `uv run python -m rag.chunking` should end with **Total chunks: 17**. Read a few chunks. Would each one make sense on its own?

✅ Checkpoint: `step-6.1`

### 6.2 Embedding and the vector store: `rag/embedding.py`

An embedding model maps text to a vector, so that texts with similar meaning land close together. We use a small local model (no API key; it downloads on first run). Our vector store is a list of chunks plus a numpy matrix with one row per chunk.

```python
def embed_documents(texts: list[str]) -> np.ndarray:
    return np.array(list(get_model().passage_embed(texts)))


def embed_query(text: str) -> np.ndarray:
    # Queries are embedded slightly differently from documents for this model.
    return np.array(list(get_model().query_embed(text)))[0]
```

```python
    def add(self, chunks: list[str]) -> None:
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, embed_documents(chunks)])
```

```python
def build_vector_store() -> VectorStore:
    """Load the service guide, chunk it, embed it, store it."""
    store = VectorStore()
    store.add(chunk_document(DOCUMENT_PATH.read_text()))
    return store
```

Check: `uv run python -m rag.embedding` should print **Vector matrix shape: (17, 384)**: 17 chunks, 384 numbers each.

✅ Checkpoint: `step-6.2`

### 6.3 Vector distance: `cosine_similarity` in `rag/retrieval.py`

Cosine similarity measures the angle between two vectors: 1.0 means the same direction (same meaning), and 0 means unrelated. Compare the query to every stored vector at once:

```python
def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
```

Check:

```bash
uv run python -c "
import numpy as np
from rag.retrieval import cosine_similarity
print(cosine_similarity(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])).round(3))"
```

You should get `[1.    0.    0.707]`: identical, unrelated, and halfway.

✅ Checkpoint: `step-6.3`

### 6.4 Retrieval: `search` in `rag/retrieval.py`

Embed the question, score every chunk, and return the top `k`:

```python
def search(store: VectorStore, query: str, k: int = 3) -> list[tuple[float, str]]:
    scores = cosine_similarity(embed_query(query), store.vectors)
    best = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), store.chunks[i]) for i in best]
```

Check:

```bash
uv run python -m rag.retrieval "what happens if I cancel late?"
```

The top result should be **## Cancellations and Changes**. The question shares almost no words with that section ("less than 4 hours before collection"), yet it ranks first. That's retrieval by meaning.

**Try it:** find a question that retrieves the *wrong* chunk. Would you fix it with better chunking, a bigger `k`, or a rewritten query?

✅ Checkpoint: `step-6.4`

---

## Step 7: The classifier and routing

**Concept:** the agent loop is **non-deterministic**: the model picks the steps. Routing doesn't need to be. We put a **deterministic** step in front: one LLM call that must return `"dispute"` or `"enquiry"` (a structured output, enforced by a schema), followed by a plain `if/else` in our code.

- **dispute** → the Dispute Agent (database tools and chat)
- **enquiry** → RAG retrieval (plain code) → an Enquiry Agent that only has `write_message` (**least privilege**: it can't read customer data)

**Edit `harness/agent.py`** so an agent can be limited to some of the server's tools:

```python
    def __init__(
        self,
        name: str,
        system_prompt: str,
        allowed_tools: list[str] | None = None,
        verbose: bool = True,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools  # None = every tool the MCP server offers
        self.verbose = verbose
```

and in `__aenter__`, filter the tool list:

```python
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in mcp_tools
            if self.allowed_tools is None or t.name in self.allowed_tools
        ]
```

**Replace all of `harness/workflow.py` with this router version:**

```python
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
```

**Add a routing eval to `evals/test_agent.py`.** Add an `"agent"` key as the first entry of each case in `CASES`, naming the agent it should reach (keep the existing keys):

```python
    "dispute_acme.txt": {
        "agent": "Dispute Agent",
        ...
    "dispute_coastal.txt": {
        "agent": "Dispute Agent",
        ...
    "enquiry_refrigerated.txt": {
        "agent": "Enquiry Agent",
        ...
    "enquiry_cancellation.txt": {
        "agent": "Enquiry Agent",
        ...
```

Then add this test above `test_agent_wrote_one_message` (it's the strictest check, so it goes first):

```python
@pytest.mark.parametrize("email", CASES)
def test_routed_to_the_right_agent(agents, email):
    assert agents[email].name == CASES[email]["agent"]
```

**Run it:**

```bash
uv run python -m harness.main data/emails/enquiry_refrigerated.txt
uv run pytest evals -v
```

**What you should see:** `[Router] category=enquiry`, then the Enquiry Agent posts an answer about the 2–8 °C range and temperature logs, citing the documentation. The evals show **20 passed**.

**Try it:**
- Write an email that sits on the boundary between the two categories, such as *"How is the heavy item surcharge calculated? I think I was charged it wrongly."* Where does it go? Should it go there?
- Run the evals three times. Does anything flake? If it does, is the agent wrong or the eval?
- Break the agent on purpose: tell the dispute prompt to "write casually, like a text message". Which evals catch it?

✅ Checkpoint: `step-7` (the finished solution)

---

## Wrap-up

Read [`docs/agentic-development.md`](docs/agentic-development.md) for how to set up AI coding agents on your own projects, and the lessons from today.
