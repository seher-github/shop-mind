"""Agent 2: Inventory/FAQ Analyst. Reads the retrieved shop data and decides the factual answer."""
import re

from crewai import Agent, Task


def create_faq_agent(llm) -> Agent:
    return Agent(
        role="Inventory and FAQ Analyst",
        goal="Decide, from the retrieved shop data only, what the correct factual answer to the customer's question is.",
        backstory=(
            "You work in a small clothing shop. You trust only the retrieved shop data, never your own memory, "
            "and you never guess prices, sizes, stock or policies."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )


def create_faq_task(agent: Agent, message: str, query: str, history: str, retrieved: str) -> Task:
    return Task(
        description=(
            "Recent conversation (may be empty):\n"
            f"{history}\n\n"
            f"Customer's latest message: {message}\n"
            f"Search query used: {query}\n\n"
            "Retrieved shop data:\n"
            f"{retrieved}\n\n"
            "Using ONLY the retrieved shop data, decide the factual answer. Copy prices, sizes, stock levels and "
            "policy details exactly. Output in this exact format:\n"
            "ANSWERABLE: YES, PARTIAL or NO\n"
            "FACTS:\n"
            "- one fact per line\n"
            "Use YES when the data fully answers the question, PARTIAL when it answers only part of it, "
            "and NO when it does not answer it or says NO_MATCH (then write only: NO_MATCH)."
        ),
        expected_output="The line 'ANSWERABLE: YES/PARTIAL/NO' followed by 'FACTS:' and a short bullet list.",
        agent=agent,
    )


def parse_faq(raw: str) -> str:
    m = re.search(r"ANSWERABLE:\s*(YES|PARTIAL|NO)", (raw or "").upper())
    return m.group(1) if m else "PARTIAL"
