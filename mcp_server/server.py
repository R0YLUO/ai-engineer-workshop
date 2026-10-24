"""MCP server exposing Swift Freight's internal systems as tools.

An MCP server is just a process that says "here are my tools" and "run this tool
for me". Any MCP client (our harness, Claude Code, Claude Desktop...) can use it.

It runs as its own process and serves MCP over HTTP:

    uv run python mcp_server/server.py        # listening on http://127.0.0.1:8001/mcp

Then poke at it from another terminal:
    uv run fastmcp list http://127.0.0.1:8001/mcp
    uv run fastmcp call http://127.0.0.1:8001/mcp lookup_customer name=acme
"""

from fastmcp import FastMCP

import db

mcp = FastMCP("swift-freight-systems")


# The docstring and type hints ARE the tool definition the model sees.
# Write them for the model: say what the tool does and when to use it.


@mcp.tool
def lookup_customer(name: str) -> list[dict]:
    """Find customers by (partial) company name.

    Returns each matching customer's id, contact email and their CONTRACTED
    pricing (rate per km by service level, discount %, surcharges).
    Use this first to get the customer_id.
    """
    return db.find_customers(name)


@mcp.tool
def get_booking_details(customer_id: str) -> list[dict]:
    """Get all bookings for a customer, including the invoice for each booking.

    Each invoice shows the values that were ACTUALLY applied (rate per km,
    surcharges, discount) and the total that was charged.
    """
    return db.get_bookings(customer_id)


@mcp.tool
def write_message(message: str) -> dict:
    """Post a message to the finance team's chat channel for human review.

    Use this to surface findings, recommended actions and draft email replies.
    A human will review the message before anything is sent to the customer.
    """
    return db.add_message(message)


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001)
