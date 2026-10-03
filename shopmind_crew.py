"""The coordinator: runs the whole ShopMind pipeline for one customer message.

 Query Rewriter -> Hybrid Retriever -> FAQ Analyst -> Reply Agent -> Verification -> Escalation Agent
"""
import os
import re
import time

# Turn off CrewAI's anonymous usage reporting (must be set BEFORE importing crewai).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Crew, Process  # noqa: E402

from agents.escalation_agent import (  # noqa: E402
    ROUTE_NAMES, create_escalation_agent, create_escalation_task, final_route, parse_route, rule_route,
)
from agents.faq_agent import create_faq_agent, create_faq_task, parse_faq  # noqa: E402
from agents.reply_agent import create_reply_agent, create_reply_task  # noqa: E402
from agents.rewriter_agent import create_rewriter_agent, create_rewriter_task, parse_rewrite  # noqa: E402
from llm import GROQ_BASE_URL, get_llm  # noqa: E402
from verification import verify  # noqa: E402

HOLD = {
    "review": {
        "english": "Thanks for your message! I've passed this to the shop owner, who will get back to you personally very soon.",
        "roman": "Shukriya! Aap ka message shop owner ko bhej diya gaya hai, woh jald aap se rabta karenge.",
    },
    "escalate": {
        "english": "I'm really sorry about this. I've alerted the shop owner right away and they will contact you personally.",
        "roman": "Is takleef ke liye hum maazrat khwah hain. Aap ka message shop owner ko foran bhej diya gaya hai, woh khud aap se rabta karenge.",
    },
}
ROMAN_HINTS = {"kya", "hai", "hain", "kitne", "kitna", "ka", "ki", "aap", "mein", "nahi", "nahin", "bhai", "karte", "chahiye",
               "mil", "milega", "kab", "kahan", "wala", "woh", "mujhe", "shikayat", "dhoka", "kharab", "ghalat"}


def guess_roman_urdu(text: str) -> bool:
    return len(set(re.findall(r"[a-z]+", text.lower())) & ROMAN_HINTS) >= 2


def _run(agent, *tasks):
    Crew(agents=[agent], tasks=list(tasks), process=Process.sequential, verbose=False).kickoff()


def _result(route_key, reason, message, language, **extra):
    route = ROUTE_NAMES[route_key]
    roman = language in ("roman_urdu", "urdu", "mixed")
    base = {
        "route": route, "needs_owner": route != "auto", "reason": reason,
        "reply": "", "facts": "", "retrieved": "", "query": message, "language": language, "intent": "other",
        "confidence": "n/a", "answerable": "n/a", "verification": {"ok": True, "unsupported": [], "checked": 0},
        "examples_used": 0, "seconds": 0.0, "trace": [], "customer_text": "",
    }
    base.update(extra)
    if route != "auto":
        base["customer_text"] = HOLD[route]["roman" if roman else "english"]
    return base


def answer_customer(message: str, history: str, api_key: str, index, base_url: str = GROQ_BASE_URL) -> dict:
    """Runs the pipeline on ONE shop's data (index = HybridIndex). Returns a dict describing everything that happened."""
    t0 = time.time()
    shop = index.shop
    trace = []

    # Safety first: obvious complaints skip the AI completely (instant, no free-tier quota used).
    rule = rule_route(message)
    guessed_lang = "roman_urdu" if guess_roman_urdu(message) else "english"
    if rule[0] == "ESCALATE":
        trace.append({"step": "Rules", "detail": rule[1]})
        res = _result("ESCALATE", f"[Rule] {rule[1]}", message, guessed_lang, intent="complaint", trace=trace)
        res["seconds"] = round(time.time() - t0, 1)
        return res

    llm = get_llm(api_key, base_url)

    # 1. Query Rewriter
    rewriter = create_rewriter_agent(llm)
    rw_task = create_rewriter_task(rewriter, message, history)
    _run(rewriter, rw_task)
    rew = parse_rewrite(rw_task.output.raw, message)
    trace.append({"step": "Rewrite", "detail": f"“{rew['query']}” · {rew['language']} · intent: {rew['intent']}"})

    # The rewritten query can reveal a complaint hidden in slang.
    rule = rule_route(message, rew["query"])
    if rule[0] == "ESCALATE":
        trace.append({"step": "Rules", "detail": rule[1]})
        res = _result("ESCALATE", f"[Rule] {rule[1]}", message, rew["language"], intent=rew["intent"], query=rew["query"], trace=trace)
        res["seconds"] = round(time.time() - t0, 1)
        return res

    # 2. Hybrid retriever (code)
    ret = index.search(rew["query"], message)
    n_p, n_l, n_c = len(ret["products"]), len(ret["policies"]), len(ret["corrections"])
    trace.append({"step": "Retrieve", "detail": f"{n_p} product(s), {n_l} policy row(s), {n_c} owner example(s) · confidence: {ret['confidence']}"})

    # 3 + 4. FAQ Analyst -> Reply Agent
    faq_agent = create_faq_agent(llm)
    reply_agent = create_reply_agent(llm, shop["shop_name"])
    faq_task = create_faq_task(faq_agent, message, rew["query"], history, ret["facts_text"])
    reply_task = create_reply_task(reply_agent, message, rew["language"], history, faq_task,
                                   shop["owner_contact"], ret["examples_text"])
    Crew(agents=[faq_agent, reply_agent], tasks=[faq_task, reply_task], process=Process.sequential, verbose=False).kickoff()
    faq_raw = faq_task.output.raw.strip()
    answerable = parse_faq(faq_raw)
    draft = reply_task.output.raw.strip()
    trace.append({"step": "FAQ", "detail": f"answerable: {answerable}"})
    trace.append({"step": "Reply", "detail": "draft written"})

    # 5. Answer verification (code)
    ver = verify(draft, ret["facts_text"], message)
    trace.append({"step": "Verify", "detail": "all claims supported" if ver["ok"] else "unsupported: " + "; ".join(ver["unsupported"][:3])})

    # 6. Escalation: rules + verification first, AI only when everything else says auto-send
    llm_result = None
    code_route, _ = final_route(rule, ver, answerable, ret["confidence"], None)
    if code_route == "AUTO_SEND":
        esc_agent = create_escalation_agent(llm)
        esc_task = create_escalation_task(esc_agent, message, draft, ret["facts_text"],
                                          "all numbers, sizes and promises are supported" if ver["ok"] else "problems found", answerable)
        _run(esc_agent, esc_task)
        llm_result = parse_route(esc_task.output.raw)
    route_key, reason = final_route(rule, ver, answerable, ret["confidence"], llm_result)
    trace.append({"step": "Route", "detail": ROUTE_NAMES[route_key]})

    res = _result(route_key, reason, message, rew["language"], intent=rew["intent"], query=rew["query"],
                  reply=draft, facts=faq_raw, retrieved=ret["facts_text"], confidence=ret["confidence"],
                  answerable=answerable, verification=ver, examples_used=n_c, trace=trace)
    if route_key == "AUTO_SEND":
        res["customer_text"] = draft
    res["seconds"] = round(time.time() - t0, 1)
    return res
