"""Agent 3: decides if the draft reply is safe to send automatically.

Two layers of safety:
1. A simple keyword check in plain Python (cannot be fooled by the AI).
2. An AI reviewer that compares the draft against the facts.
If EITHER layer says "owner", the message goes to the shop owner.
"""
import json
import re

from crewai import Agent, Task

# Topics that should always be handled by a human.
OWNER_KEYWORDS = [
    "refund", "complain", "complaint", "angry", "furious", "terrible", "worst", "scam",
    "fraud", "lawyer", "legal", "sue", "police", "damaged", "broken", "wrong item",
    "never arrived", "not received", "lost", "stolen", "discount", "bargain", "cheaper",
    "wholesale", "bulk", "custom", "allergic", "manager", "owner", "speak to",
]


def keyword_flags(message: str) -> list:
    text = message.lower()
    return [k for k in OWNER_KEYWORDS if k in text]


def create_review_agent(llm) -> Agent:
    return Agent(
        role="Reply Safety Reviewer",
        goal="Decide whether a drafted reply can be sent to the customer automatically or must be checked by the shop owner.",
        backstory=(
            "You are a careful shop supervisor. A reply is safe only if every price, size, stock "
            "level and policy in it appears in the facts, and the customer's request is a simple "
            "question. Anything about money problems, complaints, exceptions, or missing facts "
            "needs the owner."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )


def create_review_task(agent: Agent, message: str, lookup_task: Task, reply_task: Task) -> Task:
    return Task(
        description=(
            f"Customer's latest message: {message}\n\n"
            "Compare the drafted reply with the facts from the earlier tasks.\n"
            "Choose NEEDS_OWNER if ANY of these is true: the reply contains something not in the facts; "
            "the facts were NO_MATCH or incomplete; the customer is upset, complaining, asking for a refund, "
            "discount or exception; the question is unclear or risky.\n"
            "Otherwise choose AUTO_SEND.\n"
            "Answer with ONE line of JSON and nothing else, using exactly these two keys: "
            "decision (AUTO_SEND or NEEDS_OWNER) and reason (max 15 words)."
        ),
        expected_output="One line of JSON with the keys decision and reason.",
        agent=agent,
        context=[lookup_task, reply_task],
    )


def parse_decision(raw: str, message: str, facts: str) -> dict:
    """Combine the AI's verdict with the hard safety rules. Defaults to the owner if unsure."""
    needs_owner = True
    reason = "Could not read the reviewer's answer, so the owner should check."

    match = re.search(r"\{.*?\}", raw or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            decision = str(data.get("decision", "")).upper().strip()
            reason = str(data.get("reason", "")).strip() or "No reason given."
            needs_owner = decision != "AUTO_SEND"
        except json.JSONDecodeError:
            pass

    flags = keyword_flags(message)
    if flags:
        needs_owner = True
        reason = f"Sensitive topic detected ({', '.join(flags)})."
    elif "NO_MATCH" in (facts or "").upper():
        needs_owner = True
        reason = "No matching information in the shop data."

    return {"needs_owner": needs_owner, "reason": reason}
