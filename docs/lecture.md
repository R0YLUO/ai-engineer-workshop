# Lecture: Building AI, not using AI

These are speaker notes for the opening lecture. Each section maps to a module of the hands-on workshop.

---

## 1. AI as a tool vs AI as a feature

| AI as a tool | AI as a feature |
|---|---|
| *You* use ChatGPT, Claude or Copilot to do your work | *Your product* uses a model to do work for its users |
| A human is in the loop on every prompt | The prompt is written once, by you, and then runs thousands of times |
| "Good enough" is judged by eye, in the moment | "Good enough" must be **measured** (evals) |
| The context is whatever you paste in | The context is **engineered**: retrieved, filtered and assembled by code |
| The chat app owns the tools | **You** define the tools, and their permissions |

An AI engineer builds the system around the model: the **harness**, the **context**, the **tools** and the **evals**. The model is one component. Most of the engineering effort goes into everything around it.

---

## 2. The model is a function: messages in, message out

- The API is **stateless**. Every call sends the full conversation.
- "Memory", "chat" and "agents" are all built by your code managing a `messages` list.
- What the model knows on a given turn is exactly what is in that list: the **context window**. Context engineering means choosing what goes in it.
- Tokens cost money and latency. Context that keeps growing makes every turn more expensive.

➡️ Workshop: `harness/examples/01_first_call.py`

---

## 3. Tool use: how a model takes actions

- A tool is a name, a description and a JSON schema. The model never runs code. It **asks** you to run a tool by returning a `tool_use` block.
- Your code runs the tool and sends back a `tool_result`, then calls the model again.
- **An agent is a model using tools in a loop** until it decides it's done.

```
while True:
    response = model(system, tools, messages)
    messages += response
    if no tool calls: break
    messages += run(tool calls)
```

- The tool description is a prompt. Write it for the model: say what the tool does and when to use it.

➡️ Workshop: `harness/examples/02_tool_use.py` → `harness/agent.py`

---

## 4. Workflows vs agents: deterministic vs non-deterministic

| Deterministic workflow | Non-deterministic (agentic) |
|---|---|
| **Your code** decides the steps | **The model** decides the steps |
| Predictable, testable, cheap | Flexible, handles the unexpected |
| e.g. classify → if/else route | e.g. "investigate this dispute" |

Rule of thumb: **use code for the parts you can specify, and give the model the parts you can't.**

Our design is a deterministic **router**: one LLM call with a fixed output schema, followed by an `if/else` in Python. The router hands off to a specialised agent. Each agent gets only the tools it needs (least privilege). The enquiry agent can't query customer data.

➡️ Workshop: `harness/workflow.py`

---

## 5. MCP: a standard plug for tools

- Without MCP, every app re-implements its integrations to every system.
- **Model Context Protocol** gives tools a standard interface. A server exposes tools, and any client (your harness, Claude Code, Claude Desktop) can discover and call them.
- On start, our harness **connects → lists tools → converts them to Claude's tool format**. On each tool call, it **forwards the call to the server**.
- The systems team can own the MCP server, and the AI team only consumes it.

➡️ Workshop: `mcp_server/server.py` + the MCP section of `harness/agent.py`

---

## 6. RAG: giving the model knowledge it doesn't have

- The model doesn't know *your* business rules, and a 50-page manual is too big and too noisy to paste into every prompt.
- **Retrieval-Augmented Generation** works in three steps:
  1. **Chunk:** split documents into self-contained pieces.
  2. **Embed:** turn each chunk into a vector. Similar meaning gives nearby vectors.
  3. **Retrieve:** embed the question, find the nearest chunks and put only those in the prompt.
- Most RAG quality problems are **retrieval** problems: bad chunks, or the wrong chunks retrieved. Debug retrieval before you touch the prompt.

➡️ Workshop: `rag/`

---

## 7. Evals: tests for non-deterministic systems

- You can't `assert output == expected` on prose. Run the real system on known inputs and check the **properties** you care about, from strict to fuzzy:
  1. **Outcome:** did it route correctly? Did it post a message?
  2. **Trajectory:** did it call the right tools, and avoid the ones it shouldn't use?
  3. **Content:** does the output contain the right numbers?
  4. **Quality:** is the tone right? Use **LLM-as-judge** with a narrow rubric.
- Evals turn "it seemed fine when I tried it" into a number you can track whenever you change a prompt, a model or a tool.

➡️ Workshop: `evals/`

---

## 8. Human in the loop

The agent never emails the customer. It **surfaces findings and drafts a reply**, and a human approves it. Start every AI feature this way. Give the agent more autonomy only once your evals show you can trust it.
