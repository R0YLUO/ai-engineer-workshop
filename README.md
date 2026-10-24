# AI Engineer Workshop

This monorepo covers the technical pieces an AI engineer works with: the **harness**, **tool use**, **workflows**, **evals**, **MCP** and **RAG**. The boilerplate is already written, so you can spend your time on the parts you learn the most from.

The workshop is about **building AI as a feature**. It's not about using AI as a tool.

## Running the workshop

Follow [`guide.md`](guide.md). `main` is the starting state: a stub that you build into the full agent one step at a time (a single LLM call, evals, MCP, tools, the agentic loop, RAG and finally the classifier). The history of `main` holds one commit per step, building up to the finished solution, and each has a git tag (`step-0` … `step-7`) so anyone who falls behind can jump to it.

The rest of this README describes the finished system (tag `step-7`).

## The problem

The finance team at **Swift Freight Co.**, a delivery services company, invoices its customers for deliveries. The invoicing process is messy. Staff often key the wrong values into the database (wrong rate, missing discount), and customers then email to complain. For each complaint, someone has to match the booking against the contract and the invoice by hand to work out what went wrong.

Customers also email general questions, such as "what temperature do your fridge trucks hold?" The answers are buried in a long service guide that someone has to read through.

The result is slow operations and unhappy customers.

## The solution we build

The solution is an agentic workflow that triages every email. The agent investigates first: it checks the right systems, reasons about what it finds and drafts a reply. It then posts its findings to the team chat, where **a human reviews them before anything goes to the customer.**

```
                       ┌──────────────────────┐
   📧 email ─────────► │  Router (classifier) │   deterministic: 1 LLM call, fixed schema, if/else in code
                       └──────────┬───────────┘
                 "dispute"        │        "enquiry"
             ┌────────────────────┴───────────────────┐
             ▼                                        ▼
   ┌───────────────────┐                   ┌─────────────────────┐
   │   Dispute Agent   │                   │   RAG retrieval     │  chunk → embed → search
   │   (agent loop)    │                   │   over service guide│
   └────────┬──────────┘                   └──────────┬──────────┘
            │ MCP tools                               ▼
            │ lookup_customer                ┌─────────────────────┐
            │ get_booking_details            │   Enquiry Agent     │
            │ write_message                  │   (agent loop)      │
            │                                └──────────┬──────────┘
            ▼                                           │ MCP tool: write_message
   ┌────────────────────────────────────────────────────▼─────────┐
   │  MCP server (HTTP :8001): in-memory DB + chat (messages.json)│
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
                   💬 Chat app (polls every 2s): 👩‍💼 human reviews
```

## Repo map

| Folder | What's in it | Key concept |
|---|---|---|
| [`harness/examples/`](harness/examples) | `01_first_call.py` and `02_tool_use.py`, two short warm-up scripts | Messages API, stateless calls, the tool-use loop |
| [`harness/agent.py`](harness/agent.py) | The `Agent` class: connects to MCP, discovers tools, runs the loop, logs the growing context | **The harness** |
| [`harness/workflow.py`](harness/workflow.py) | The classifier/router and the two agents' prompts | **Deterministic vs non-deterministic workflows**, multi-agent routing |
| [`harness/main.py`](harness/main.py) | Runs the workflow on one email | |
| [`mcp_server/`](mcp_server) | FastMCP server, in-memory DB and the chat web app | **MCP** |
| [`rag/`](rag) | `chunking.py`, `embedding.py` (vector store) and `retrieval.py` | **RAG** |
| [`evals/`](evals) | pytest evals and an LLM-as-judge | **Evals** |
| [`data/`](data) | Sample emails and the service guide document | |
| [`guide.md`](guide.md) | The step-by-step workshop guide | |
| [`docs/`](docs) | Lecture notes and the agentic development wrap-up | |

## Setup

You need [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
uv sync                      # install dependencies
cp .env.example .env         # then paste your ANTHROPIC_API_KEY into .env
```

## Run the demo

The demo uses three terminals, all run from the repo root.

**Terminal 1** starts the MCP server. It runs as its own process and listens on `http://127.0.0.1:8001/mcp`:

```bash
uv run python mcp_server/server.py
```

**Terminal 2** starts the chat app, a static page that reads `messages.json`:

```bash
uv run python -m http.server 8000 -d mcp_server/chat_app
# open http://localhost:8000
```

**Terminal 3** runs the agent on an email. The agent connects to the MCP server over HTTP:

```bash
uv run python -m harness.main data/emails/dispute_acme.txt
uv run python -m harness.main data/emails/dispute_coastal.txt
uv run python -m harness.main data/emails/enquiry_refrigerated.txt
uv run python -m harness.main data/emails/enquiry_cancellation.txt
```

The terminal shows the context growing turn by turn and every tool call. The agent's findings appear in the chat app.

If you see `Client failed to connect`, the MCP server in terminal 1 isn't running.

To clear the chat: `rm mcp_server/chat_app/messages.json`

## The MCP server

`mcp_server/server.py` is a standalone process that serves MCP over HTTP (the streamable HTTP transport) at `http://127.0.0.1:8001/mcp`. Clients connect to that URL. The harness does this in `Agent.__aenter__` (`harness/agent.py`): it connects, lists the tools and forwards Claude's tool calls to the server.

With the server running, you can talk to it directly from another terminal:

```bash
# List the tools exactly as an MCP client (and Claude) sees them
uv run fastmcp list http://127.0.0.1:8001/mcp

# Call a tool from the command line
uv run fastmcp call http://127.0.0.1:8001/mcp lookup_customer name=acme
uv run fastmcp call http://127.0.0.1:8001/mcp get_booking_details customer_id=C001
uv run fastmcp call http://127.0.0.1:8001/mcp write_message message="Hello from the CLI"   # shows up in the chat app
```

Any MCP client can connect to the same URL. For example, to add it to Claude Code:

```bash
claude mcp add --transport http swift-freight http://127.0.0.1:8001/mcp
```

The database lives in the server process's memory. It's shared by every client and resets when you restart the server.

## Try each piece on its own

```bash
uv run python harness/examples/01_first_call.py      # one API call; the context is resent each time
uv run python harness/examples/02_tool_use.py        # a tool and the agentic loop, in 60 lines
uv run python -m rag.chunking                        # see the chunks
uv run python -m rag.embedding                       # see the vectors
uv run python -m rag.retrieval "can you carry gas bottles?"
uv run fastmcp list http://127.0.0.1:8001/mcp        # the MCP tools (server must be running)
uv run pytest evals -v                               # run the evals (server must be running; about 20 LLM calls)
```

## The data (answers)

The emails in `data/emails/` map to the invoice errors in `mcp_server/db.py`:

| Email | What the agent should find |
|---|---|
| `dispute_acme.txt` | INV-1042 used the **express** rate ($4.00/km) on a **standard** booking. 120 km × $2.50 = **$300.00**, not $480.00 |
| `dispute_coastal.txt` | INV-1051 left out the contracted **10% discount**. (80 × $3.10 + $40 heavy item) × 0.9 = **$259.20**, not $288.00 |
| `enquiry_refrigerated.txt` | Answered from the service guide: 2–8 °C range, temperature logs available, pricing is the base charge plus the refrigerated surcharge |
| `enquiry_cancellation.txt` | Answered from the service guide: free cancellation more than 4 hours before collection, otherwise $35. Aerosols and small gas cylinders are accepted |

Next, read [`docs/lecture.md`](docs/lecture.md), then work through [`guide.md`](guide.md).
