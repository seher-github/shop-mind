"""Agent 3: decides if the draft reply is safe to send automatically.

Two layers of safety:
1. A simple keyword check in plain Python (cannot be fooled by the AI).
2. An AI reviewer that compares the draft against the facts.
If EITHER layer says "owner", the message goes to the shop owner.
The reason always says WHICH layer decided, so you can see why.
"""
import json
import re

from crewai import Agent, Task

# Topics that should always be handled by a human. Matched as whole words
# (so "issue" does NOT trigger "sue", and "customer" does NOT trigger "custom").
OWNER_KEYWORDS = [
    "refund", "complain", "complaint", "angry", "furious", "terrible", "worst", "scam",
    "fraud", "lawyer", "legal", "sue", "police", "damaged", "broken", "wrong item",
    "never arrived", "not received", "lost", "stolen", "discount", "bargain", "cheaper",
    "wholesale", "bulk", "custom order", "custom made", "customise", "customize", "allergic",
    "manager", "owner", "speak to",
]
_KEYWORD_PATTERNS = {
    k: re.compile(r"\b" + re.escape(k) + r"(?:s|d|ed|ing)?\b") for k in OWNER_KEYWORDS
}


def keyword_flags(message: str) -> list:
    text = message.lower()
    return [k for k, pat in _KEYWORD_PATTERNS.items() if pat.search(text)]


def create_review_agent(llm) -> Agent:
    return Agent(
        role="Reply Safety Reviewer",
        goal="Decide whether a drafted reply can be sent to the customer automatically or must be checked by the shop owner.",
        backstory=(
            "You are a fair shop supervisor. Ordinary questions about prices, sizes, stock, delivery, "
            "returns and opening hours that are answered correctly from the shop's facts are safe to send. "
            "You only involve the owner when the reply invents something, the facts do not cover the "
            "question, or the customer is upset or asking for an exception."
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
            "Default to AUTO_SEND. A reply is supported when every price, size, stock level, delivery time "
            "or policy it states appears in the facts. Different wording is fine, and a reply that uses only "
            "part of the facts is fine.\n"
            "Choose NEEDS_OWNER only if ONE of these is true:\n"
            "1. the reply states something that is NOT in the facts;\n"
            "2. the facts are NO_MATCH or clearly do not cover the question;\n"
            "3. the customer is upset, complaining, or asking for a refund, discount or exception;\n"
            "4. the message is unclear or asks for something risky.\n"
            "Answer with ONE line of JSON and nothing else, using exactly these two keys: "
            "decision (AUTO_SEND or NEEDS_OWNER) and reason (max 15 words)."
        ),
        expected_output="One line of JSON with the keys decision and reason.",
        agent=agent,
        context=[lookup_task, reply_task],
    )


def _compact(text: str) -> str:
    """'auto-send', 'AUTO SEND.' and 'AUTO_SEND' all become 'AUTOSEND'."""
    return re.sub(r"[^A-Z]", "", str(text).upper())


def parse_decision(raw: str, message: str, facts: str) -> dict:
    """Combine the AI's verdict with the hard safety rules. Defaults to the owner if unsure."""
    text = raw or ""
    decision, reason = "", ""

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            decision = _compact(data.get("decision", ""))
            reason = str(data.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass
    if "AUTOSEND" not in decision and "NEEDSOWNER" not in decision:
        compact = _compact(text)          # fall back to finding the words anywhere
        decision = "NEEDSOWNER" if "NEEDSOWNER" in compact else ("AUTOSEND" if "AUTOSEND" in compact else "")

    if "NEEDSOWNER" in decision:
        needs_owner, why = True, f"[AI reviewer] {reason or 'Owner should check.'}"
    elif "AUTOSEND" in decision:
        needs_owner, why = False, f"[AI reviewer] {reason or 'Matches the facts.'}"
    else:
        needs_owner, why = True, "[Unreadable reviewer answer] The AI reviewer's reply could not be understood."

    flags = keyword_flags(message)
    if flags:
        needs_owner, why = True, f"[Safety keyword] Sensitive topic detected ({', '.join(flags)})."
    elif "NO_MATCH" in (facts or "").upper():
        needs_owner, why = True, "[No matching data] Nothing in the shop data matched this question."

    return {"needs_owner": needs_owner, "reason": why}
