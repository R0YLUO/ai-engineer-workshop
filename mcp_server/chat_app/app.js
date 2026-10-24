// Polls messages.json (written by the MCP server's write_message tool) every 2 seconds.

const container = document.getElementById("messages");
let shownCount = 0;

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function render(messages) {
  container.innerHTML = messages.map((message) => `
    <div class="message">
      <div class="meta"><strong>${escapeHtml(message.author)}</strong> · ${escapeHtml(message.timestamp)}</div>
      <div class="text">${escapeHtml(message.text)}</div>
    </div>`).join("");
  window.scrollTo(0, document.body.scrollHeight);
}

async function poll() {
  try {
    // ?t=... stops the browser from caching the file
    const response = await fetch(`messages.json?t=${Date.now()}`);
    if (!response.ok) return; // no messages.json until the agent posts its first message
    const messages = await response.json();
    if (messages.length !== shownCount) {
      render(messages);
      shownCount = messages.length;
    }
  } catch (error) {
    console.error("Could not load messages", error);
  }
}

poll();
setInterval(poll, 2000);
