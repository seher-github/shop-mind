"""Extra look-and-feel pieces for the new pipeline (the base CSS stays in ui_styles.py)."""
import html

EXTRA_CSS = """
.verdict.review { background:rgba(251,191,36,.2); border:1px solid rgba(251,191,36,.8); }
.verdict.esc { background:rgba(239,68,68,.22); border:1px solid rgba(239,68,68,.85); }
.step.skip { opacity:.35; text-decoration:line-through; }
ul.trace { margin:.2rem 0 .6rem 0; padding-left:1.1rem; font-size:.86rem; color:#d6d6f5; }
ul.trace li { margin-bottom:.15rem; }
.agent-row { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; }
"""

ROUTE_BADGE = {
    "auto": '<span class="verdict ok">⚡ Auto-sent</span>',
    "review": '<span class="verdict review">📝 Needs review</span>',
    "escalate": '<span class="verdict esc">🚨 Escalated to owner</span>',
}
PIPELINE = [("Rewrite", "✍️"), ("Retrieve", "🔎"), ("FAQ", "🧠"), ("Reply", "💬"), ("Verify", "✅"), ("Route", "🛡️")]


def hero_html(shop_name: str, source_label: str) -> str:
    chips = "".join(f'<span class="chip">{c}</span>' for c in
                    ["✍️ Rewriter", "🔎 Hybrid retriever", "🧠 FAQ agent", "💬 Reply agent", "✅ Verification", "🛡️ Escalation"])
    return (
        '<div class="hero">'
        '<div class="hero-badge">✨ Multi-agent RAG · CrewAI × Groq × ChromaDB</div>'
        '<h1 class="hero-title">Shop<span>Mind</span></h1>'
        f'<p class="hero-sub">Instant, verified answers for <b>{html.escape(shop_name)}</b> customers in English and Roman Urdu. '
        'The owner steps in only when it matters, and ShopMind learns from every correction.</p>'
        f'<div class="chips"><span class="chip live">● {html.escape(source_label)}</span></div>'
        f'<div class="agent-row">{chips}</div></div>'
    )


def route_badge(route: str) -> str:
    return ROUTE_BADGE[route]


def pipeline_html(result: dict) -> str:
    done = {t["step"] for t in result.get("trace", [])}
    steps = []
    if "Rules" in done:
        steps.append('<span class="step">⚖️ Rules</span><span class="arrow">→</span>')
    for i, (name, icon) in enumerate(PIPELINE):
        cls = "step" if name in done else "step skip"
        steps.append(f'<span class="{cls}">{icon} {name}</span>')
        steps.append('<span class="arrow">→</span>')
    row = '<div class="steps">' + "".join(steps) + ROUTE_BADGE[result["route"]] + "</div>"
    items = "".join(
        f'<li><b>{html.escape(t["step"])}</b> - {html.escape(t["detail"])}</li>' for t in result.get("trace", [])
    )
    return row + f'<ul class="trace">{items}</ul>'
