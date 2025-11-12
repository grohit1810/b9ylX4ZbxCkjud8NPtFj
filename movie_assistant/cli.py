# movie_assistant/cli.py
import asyncio
from uuid import uuid4

from movie_assistant.agent import MovieAgent
from config.settings import MCP_SERVER_CONFIG, LLM_CONFIG

BANNER = r"""
╔════════════════════════════════════════════════════════════════╗
║          MOVIE RECOMMENDATION ASSISTANT (ReAct)                ║
║                                                                ║
║  Powered by: Ollama + MCP + ReAct Agent                        ║
║                                                                ║
║  Commands:                                                     ║
║  - Ask about movies (genres, actors, themes, plots)            ║
║  - 'quit' or 'q' to exit                                       ║
║  - 'reset' to clear conversation history                       ║
║  - 'history' to show conversation                              ║
║  - 'debug on/off' to toggle thought visibility                 ║
╚════════════════════════════════════════════════════════════════╝
"""

async def run_cli():
    print(BANNER)
    print("🚀 Initializing MovieAgent…")
    mcp_url = f"http://{MCP_SERVER_CONFIG['mcp_server_host']}:{MCP_SERVER_CONFIG['mcp_server_port']}{MCP_SERVER_CONFIG['mcp_http_path']}"
    agent = MovieAgent(llm_memory_db=LLM_CONFIG["conversation_checkpoint_db"],mcp_url=mcp_url,
                       llm_host=LLM_CONFIG['host'],
                       llm_model=LLM_CONFIG['model'],
                       temperature=LLM_CONFIG['temperature'],
                       verbose=True)
    await agent._load_mcp_tools()
    print(f"MCP Tools loaded: {', '.join(agent.tool_names())}")
    print("\nType 'reset' to clear context, 'history' to print, 'quit' to exit.\n")
    thread_id = str(uuid4())
    while True:
        try:
            text = input("You: ").strip()

            if not text:
                continue

            low = text.lower()
            if low in {"quit", "q", "exit"}:
                print("Bye!")
                return
            if low == "history":
                hist = await agent.ahistory(thread_id)
                if not hist:
                    print("∅ No conversation yet.\n")
                    continue
                print("\n📜 Conversation History")
                print("─" * 64)
                for m in hist:
                    who = "You" if m["role"] == "user" else "Assistant"
                    print(f"{who}: {m['content']}\n")
                print("─" * 64)
                continue
            if low == "reset":
                thread_id = str(uuid4())
                print("Conversation history cleared.\n")
                continue

            reply = await agent.answer(text, thread_id)
            print(f"\nAssistant: {reply}\n")

        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

if __name__ == "__main__":
    asyncio.run(run_cli())
