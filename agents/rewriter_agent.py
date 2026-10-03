"""Agent 1: Query Rewriter. Turns a messy customer message into a clean English search query."""
import json
import re

from crewai import Agent, Task

LANGUAGES = {"english", "roman_urdu", "urdu", "mixed"}
INTENTS = {"price", "stock", "size", "delivery", "returns", "payment", "hours", "custom_order", "bulk_order", "complaint", "other"}


def create_rewriter_agent(llm) -> Agent:
    return Agent(
        role="Query Rewriter",
        goal="Turn the customer's message into a short, clean English search query for a clothing shop's database.",
        backstory=(
            "You understand English, Urdu and Roman Urdu (Urdu written in English letters), slang, typos "
            "and vague references like 'woh wala'. You never answer the customer; you only rewrite."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )


def create_rewriter_task(agent: Agent, message: str, history: str) -> Task:
    return Task(
        description=(
            "Recent conversation (may be empty):\n"
            f"{history}\n\n"
            f"Customer's latest message: {message}\n\n"
            "Rewrite the latest message as a short English search query (3 to 8 words) containing the product, "
            "colour, size or policy topic being asked about. Use the recent conversation to resolve words like "
            "'woh', 'it' or 'that one'. Do NOT answer the question.\n"
            "Answer with ONE line of JSON and nothing else, with exactly three keys: "
            "query (the English search query), "
            "language (one of: english, roman_urdu, urdu, mixed), "
            "intent (one of: price, stock, size, delivery, returns, payment, hours, custom_order, bulk_order, complaint, other)."
        ),
        expected_output="One line of JSON with the keys query, language and intent.",
        agent=agent,
    )


def parse_rewrite(raw: str, message: str) -> dict:
    result = {"query": message, "language": "unknown", "intent": "other"}
    match = re.search(r"\{.*?\}", raw or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            q = str(data.get("query", "")).strip()
            if q:
                result["query"] = q
            lang = str(data.get("language", "")).strip().lower()
            if lang in LANGUAGES:
                result["language"] = lang
            intent = str(data.get("intent", "")).strip().lower()
            result["intent"] = intent if intent in INTENTS else "other"
        except json.JSONDecodeError:
            pass
    return result
