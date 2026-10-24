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


class Agent:
    def __init__(self, name: str, system_prompt: str, verbose: bool = True):
        self.name = name
        self.system_prompt = system_prompt
        self.verbose = verbose

        self.client = anthropic.AsyncAnthropic()
        self.mcp = Client(MCP_SERVER_URL)
        self.tools: list[dict] = []
        self.messages: list[dict] = []  # the context window
        self.tool_calls: list[dict] = []  # a log of every tool call, used by the evals

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
