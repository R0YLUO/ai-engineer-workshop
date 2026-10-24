"""The email workflow: what happens when an email arrives.

harness/main.py reads an email file and calls handle_email(). Building it out
is the workshop: follow guide.md.
"""


async def handle_email(email: str, verbose: bool = True):
    """Handle one incoming email and return the agent that handled it."""
    raise NotImplementedError("Step 1: build the agent - see guide.md")
