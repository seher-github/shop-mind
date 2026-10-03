"""Agent 4: Escalation Agent. Chooses one of three routes for every message:

  AUTO_SEND     -> the reply is sent to the customer
  NEEDS_REVIEW  -> the owner sees the draft + reason and approves or edits it
  ESCALATE      -> the owner is notified and NO reply is sent automatically (complaints etc.)

Plain-code rules run first (they cannot be fooled by the AI). The AI only runs when the rules say
"auto-send", and it can only make the decision stricter, never looser.
"""
import json
import re

from crewai import Agent, Task

SEVERITY = {"AUTO_SEND": 0, "NEEDS_REVIEW": 1, "ESCALATE": 2}
ROUTE_NAMES = {"AUTO_SEND": "auto", "NEEDS_REVIEW": "review", "ESCALATE": "escalate"}

# Complaints, anger, legal threats, problems with an order -> the owner must handle it personally.
ESCALATE_KEYWORDS = [
    "angry", "furious", "terrible", "worst", "scam", "fraud", "lawyer", "legal", "sue", "police", "damaged",
    "broken", "wrong item", "never arrived", "not received", "lost", "stolen", "complain", "complaint",
    "cheated", "disgusting", "unacceptable", "harassed",
    # Roman Urdu (extend this list with the words your customers use)
    "shikayat", "dhoka", "dhokha", "bakwas", "kharab", "ghalat", "nahi mila", "nahin mila", "nahi aaya", "nahin aaya", "loot",
]
# Business decisions -> the owner reviews the draft first.
REVIEW_KEYWORDS = [
    "refund", "discount", "bargain", "cheaper", "negotiate", "wholesale", "bulk", "custom order", "custom made",
    "customise", "customize", "allergic", "manager", "owner", "speak to",
    "rate kam", "price kam", "sasta", "sasti",
]


def _patterns(words):
    return {k: re.compile(r"\b" + re.escape(k) + r"(?:s|d|ed|ing)?\b") for k in words}


_ESC = _patterns(ESCALATE_KEYWORDS)
_REV = _patterns(REVIEW_KEYWORDS)


def rule_route(*texts: str):
    """Keyword rules. Returns (route, reason). Whole-word matching ('issue' does not trigger 'sue')."""
    text = " ".join(t.lower() for t in texts if t)
    hits = [k for k, p in _ESC.items() if p.search(text)]
    if hits:
        return "ESCALATE", f"Complaint or serious problem detected ({', '.join(hits)})."
    hits = [k for k, p in _REV.items() if p.search(text)]
    if hits:
        return "NEEDS_REVIEW", f"Business decision needed ({', '.join(hits)})."
    return "AUTO_SEND", ""


def create_escalation_agent(llm) -> Agent:
    return Agent(
        role="Escalation Manager",
        goal="Decide whether a drafted reply can be sent automatically, needs the owner's review, or must be escalated.",
        backstory=(
            "You are a fair shop supervisor. Ordinary questions about prices, sizes, stock, delivery, returns "
            "and opening hours that are answered correctly from the shop's facts are safe to send. "
            "You involve the owner only when the reply or the situation calls for it."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )


def create_escalation_task(agent: Agent, message: str, draft: str, facts: str, verification_note: str, answerable: str) -> Task:
    return Task(
        description=(
            f"Customer's latest message: {message}\n"
            f"Draft reply: {draft}\n"
            f"Facts the draft must be based on:\n{facts}\n"
            f"Automatic verification: {verification_note}\n"
            f"FAQ analyst said the data answers the question: {answerable}\n\n"
            "Default to AUTO_SEND. The draft is fine when every price, size, stock level, delivery time or policy in it "
            "appears in the facts (different wording is fine).\n"
            "Choose NEEDS_REVIEW if the draft states something not in the facts, ignores part of the question, "
            "or the request is unclear or risky.\n"
            "Choose ESCALATE only if the customer is upset, complaining, or reporting a problem with an order.\n"
            "Answer with ONE line of JSON and nothing else, with exactly two keys: "
            "route (AUTO_SEND, NEEDS_REVIEW or ESCALATE) and reason (max 15 words)."
        ),
        expected_output="One line of JSON with the keys route and reason.",
        agent=agent,
    )


def _compact(text) -> str:
    return re.sub(r"[^A-Z]", "", str(text).upper())


def parse_route(raw: str):
    """Returns (route or None, reason). None means the answer could not be read."""
    text = raw or ""
    route, reason = "", ""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            route = _compact(data.get("route", ""))
            reason = str(data.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass
    if route not in ("AUTOSEND", "NEEDSREVIEW", "ESCALATE"):
        compact = _compact(text)
        route = next((r for r in ("ESCALATE", "NEEDSREVIEW", "AUTOSEND") if r in compact), "")
    mapping = {"AUTOSEND": "AUTO_SEND", "NEEDSREVIEW": "NEEDS_REVIEW", "ESCALATE": "ESCALATE"}
    return mapping.get(route), reason


def final_route(rule, verification, answerable, confidence, llm_result):
    """Combine every layer; the strictest decision wins. Returns (route, reason)."""
    candidates = []
    r_route, r_reason = rule
    if r_route != "AUTO_SEND":
        candidates.append((r_route, f"[Rule] {r_reason}"))
    if verification and not verification["ok"]:
        candidates.append(("NEEDS_REVIEW", "[Verification] " + "; ".join(verification["unsupported"][:3])))
    if confidence == "none" or answerable == "NO":
        candidates.append(("NEEDS_REVIEW", "[No matching data] The shop data does not answer this question."))
    if llm_result is not None:
        l_route, l_reason = llm_result
        if l_route is None:
            candidates.append(("NEEDS_REVIEW", "[Unreadable AI answer] The escalation agent's reply could not be understood."))
        elif l_route != "AUTO_SEND":
            candidates.append((l_route, f"[AI escalation agent] {l_reason or 'Owner should look at this.'}"))
        else:
            ok_reason = f"[AI escalation agent] {l_reason or 'Reply matches the facts.'}"
            if not candidates:
                return "AUTO_SEND", ok_reason
    if not candidates:
        return "AUTO_SEND", "All checks passed."
    return max(candidates, key=lambda c: SEVERITY[c[0]])
