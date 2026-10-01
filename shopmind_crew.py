"""The coordinator: builds the three agents and runs them one after another."""
import os

# Turn off CrewAI's anonymous usage reporting (must be set BEFORE importing crewai).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Crew, Process  # noqa: E402

from agents.lookup_agent import create_lookup_agent, create_lookup_task  # noqa: E402
from agents.reply_agent import create_reply_agent, create_reply_task  # noqa: E402
from agents.review_agent import create_review_agent, create_review_task, parse_decision  # noqa: E402
from llm import GROQ_BASE_URL, get_llm  # noqa: E402
from tools.shop_data_tool import load_shop  # noqa: E402

OWNER_HOLD_MESSAGE = (
    "Thanks for your message! I've passed this to the shop owner, who will get back to you personally very soon."
)


def answer_customer(message: str, history: str, api_key: str, base_url: str = GROQ_BASE_URL) -> dict:
    """Runs the 3 agents. Returns facts, draft reply, needs_owner, reason, customer_text."""
    shop = load_shop()
    llm = get_llm(api_key, base_url)

    lookup_agent = create_lookup_agent(llm)
    reply_agent = create_reply_agent(llm, shop["shop_name"])
    review_agent = create_review_agent(llm)

    lookup_task = create_lookup_task(lookup_agent, message, history)
    reply_task = create_reply_task(reply_agent, message, history, lookup_task, shop["owner_contact"])
    review_task = create_review_task(review_agent, message, lookup_task, reply_task)

    crew = Crew(
        agents=[lookup_agent, reply_agent, review_agent],
        tasks=[lookup_task, reply_task, review_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()

    facts = lookup_task.output.raw.strip()
    draft = reply_task.output.raw.strip()
    verdict = parse_decision(review_task.output.raw, message, facts)

    return {
        "facts": facts,
        "reply": draft,
        "needs_owner": verdict["needs_owner"],
        "reason": verdict["reason"],
        "customer_text": OWNER_HOLD_MESSAGE if verdict["needs_owner"] else draft,
    }
