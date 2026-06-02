import os

from dotenv import load_dotenv
from graph import create_research_graph
from langchain_core.messages import HumanMessage
from core.memanto_tools import MemantoManager, create_memanto_tools


def run_session(agent_id, user_id, task, thread_id):
    load_dotenv()

    api_key = os.getenv("MOORCHEH_API_KEY")
    if not api_key:
        print("Please set MOORCHEH_API_KEY in .env")
        return

    manager = MemantoManager(api_key=api_key)
    client = manager.setup_agent(
        agent_id=agent_id,
        description="LangGraph Research Assistant with Long-Term Memory",
    )

    tools = create_memanto_tools(client, agent_id)
    app = create_research_graph("gpt-4o", tools)

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- Session for {user_id} (Thread: {thread_id}) ---")
    print(f"Task: {task}")

    events = app.stream(
        {"messages": [HumanMessage(content=task)]}, config, stream_mode="values"
    )

    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


def main():
    agent_id = "langgraph-researcher-001"
    user_id = "user-123"

    task1 = (
        "Research the top 3 benefits of using Memanto for AI agents. "
        "Store each benefit as a 'fact' in your memory with tags 'memanto,benefits'. "
        "Be specific and concise."
    )

    run_session(agent_id, user_id, task1, thread_id="session-1")

    task2 = (
        "I'm writing a report about Memanto. What were those 3 benefits you found earlier? "
        "Use your memory to recall them."
    )

    run_session(agent_id, user_id, task2, thread_id="session-2")


if __name__ == "__main__":
    main()
