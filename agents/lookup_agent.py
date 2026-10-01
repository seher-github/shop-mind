"""Agent 1: finds the facts in the shop's data. It never writes to the customer."""
from crewai import Agent, Task

from tools.shop_data_tool import shop_data_lookup


def create_lookup_agent(llm) -> Agent:
    return Agent(
        role="Shop Data Specialist",
        goal="Find the exact product and policy facts needed to answer a customer's question.",
        backstory=(
            "You work in a small clothing shop. You only trust the shop's data lookup tool, "
            "never your own memory, and you never guess prices, sizes, stock or policies."
        ),
        tools=[shop_data_lookup],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=4,
    )


def create_lookup_task(agent: Agent, message: str, history: str) -> Task:
    return Task(
        description=(
            "Recent conversation (may be empty):\n"
            f"{history}\n\n"
            f"Customer's latest message: {message}\n\n"
            "Use the Shop Data Lookup tool (one or two short searches, e.g. the product name or "
            "'return policy') to find every fact needed to answer. "
            "Then list the facts EXACTLY as the tool returned them. Do not add anything."
        ),
        expected_output=(
            "A short bullet list of the relevant facts copied from the tool. "
            "If the tool returned NO_MATCH, output only the word NO_MATCH followed by one short sentence."
        ),
        agent=agent,
    )
