"""Run the full workflow on one email.

    uv run python -m harness.main data/emails/dispute_acme.txt

Open the chat app (see README) to watch the agent's message arrive.
"""

import asyncio
import sys
from pathlib import Path

from harness.workflow import handle_email


async def main(email_path: str) -> None:
    email = Path(email_path).read_text()
    print(f"📧 Incoming email ({email_path})\n{email}")
    await handle_email(email)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "data/emails/dispute_acme.txt"))
