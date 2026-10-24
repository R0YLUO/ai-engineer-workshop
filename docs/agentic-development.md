# Final: Agentic development setup and thoughts

Today you **built** an agent. This section covers the other side: **using** agents (Claude Code, Cursor and similar tools) to build software, and how to set that up well.

## A good setup

**1. Give the agent a map: `CLAUDE.md` / `AGENTS.md`**

A short file at the repo root. The agent reads it at the start of every session.

```markdown
# Project
Email triage agent for Swift Freight finance. Python 3.12, uv.

# Commands
- Install: uv sync
- Run:     uv run python -m harness.main data/emails/dispute_acme.txt
- Evals:   uv run pytest evals -v

# Conventions
- Tools live in mcp_server/server.py; docstrings are the tool descriptions.
- Never hardcode API keys; use .env.
```

Keep it short and specific. Everything in it costs context on every task.

**2. Give it a way to check its own work**

An agent that can run tests can fix its own mistakes. One that can't will hand those mistakes to you. Fast tests, a linter and the evals in this repo are the agent's feedback loop.

**3. Give it tools through MCP**

The MCP server you wrote today plugs straight into Claude Code (`claude mcp add ...`). The same protocol serves your product and your dev workflow.

**4. Keep humans at the decision points**

Review plans before the agent implements them, and review diffs before merging. This is the same human-in-the-loop principle as the finance chat.

## Thoughts to leave with

- **The model is the easy part.** Most of the engineering is context, tools, guardrails and evals.
- **Start deterministic, add autonomy where it pays.** Most workflows should be "code with an LLM call in the middle", not "an agent in charge".
- **Context is a budget.** Every token you add costs money, adds latency and can distract the model. Retrieve narrowly and give each agent only the tools it needs.
- **If you can't measure it, you can't ship it.** Write the eval before you tune the prompt.
- **Design for the human reviewer.** Structured, cited, skimmable output is what makes human-in-the-loop fast enough to be worth it.
- **Everything in this repo transfers.** Swap the in-memory DB for Postgres, the chat file for Teams or Slack, and the local embeddings for a vector DB. The architecture stays the same.
