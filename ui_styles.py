"""Look and feel: CSS + tiny HTML helpers. Every piece of dynamic text is escaped."""
import html

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root { --violet:#8b5cf6; --pink:#ec4899; --cyan:#22d3ee; --card:rgba(255,255,255,.06); --stroke:rgba(255,255,255,.13); }
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0b1a; }
.stApp::before {
  content:""; position:fixed; inset:-25%; z-index:0; pointer-events:none; opacity:.55;
  background: conic-gradient(from 0deg at 50% 50%, rgba(139,92,246,.45), rgba(236,72,153,.35), rgba(34,211,238,.35), rgba(139,92,246,.45));
  filter: blur(110px); animation: swirl 45s linear infinite;
}
@keyframes swirl { to { transform: rotate(360deg); } }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1100px; padding-top: 1.5rem; position: relative; z-index: 1; }
[data-testid="stSidebar"] { background: rgba(16,16,36,.82); border-right: 1px solid var(--stroke); backdrop-filter: blur(16px); }
[data-testid="stSidebar"] * { z-index: 2; }

/* hero */
.hero { padding: 1.6rem 1.8rem; border-radius: 26px; border: 1px solid var(--stroke); margin-bottom: 1rem;
  background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(236,72,153,.14) 55%, rgba(34,211,238,.14)); backdrop-filter: blur(12px); }
.hero-badge { display:inline-block; font-size:.78rem; font-weight:600; letter-spacing:.04em; padding:.3rem .8rem; border-radius:999px;
  background: rgba(255,255,255,.1); border:1px solid var(--stroke); }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:3.2rem; font-weight:700; margin:.6rem 0 .1rem 0; line-height:1.05; }
.hero-title span { background: linear-gradient(90deg, var(--violet), var(--pink), var(--cyan)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { color:#cfcff0; font-size:1.02rem; margin:.2rem 0 .8rem 0; max-width: 46rem; }
.chips { display:flex; flex-wrap:wrap; gap:.5rem; }
.chip { font-size:.8rem; padding:.28rem .75rem; border-radius:999px; background:rgba(255,255,255,.08); border:1px solid var(--stroke); }
.chip.live { border-color: rgba(34,211,238,.6); box-shadow: 0 0 14px rgba(34,211,238,.35); }

/* stat cards */
.stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:.8rem; margin-bottom:1.1rem; }
.stat { padding:.9rem 1rem; border-radius:18px; background:var(--card); border:1px solid var(--stroke); transition: transform .2s, box-shadow .2s; }
.stat:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(139,92,246,.35); }
.stat .num { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; }
.stat .lbl { color:#b9b9dd; font-size:.8rem; }
@media (max-width: 800px) { .stats { grid-template-columns: repeat(2, 1fr); } .hero-title { font-size:2.4rem; } }

/* chat */
[data-testid="stChatMessage"] { background: var(--card); border:1px solid var(--stroke); border-radius:20px;
  padding:1rem 1.2rem; margin-bottom:.8rem; backdrop-filter: blur(10px); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, rgba(139,92,246,.38), rgba(236,72,153,.26)); }
[data-testid="stChatInput"] { border-radius: 999px; border:1px solid var(--stroke); box-shadow: 0 0 28px rgba(139,92,246,.25); }

/* buttons + containers */
.stButton > button, .stDownloadButton > button { border-radius:999px; border:1px solid var(--stroke); background:var(--card); color:#fff; transition: all .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--violet); box-shadow:0 0 22px rgba(139,92,246,.5); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, var(--violet), var(--pink)); border:none; }
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }

/* behind-the-scenes */
.steps { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; margin:.2rem 0 .8rem 0; }
.step { padding:.28rem .7rem; border-radius:999px; background:rgba(255,255,255,.09); border:1px solid var(--stroke); font-size:.82rem; }
.arrow { opacity:.6; }
.verdict { padding:.28rem .8rem; border-radius:999px; font-size:.82rem; font-weight:600; }
.verdict.ok { background:rgba(34,197,94,.2); border:1px solid rgba(34,197,94,.7); }
.verdict.owner { background:rgba(251,146,60,.2); border:1px solid rgba(251,146,60,.8); }
.brand { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; margin-bottom:.4rem; }
.welcome { text-align:center; padding:1.2rem 0 .4rem 0; color:#cfcff0; }
"""


def hero_html(shop_name: str, source_label: str) -> str:
    return (
        '<div class="hero">'
        '<div class="hero-badge">✨ Multi-agent AI · CrewAI × Groq</div>'
        '<h1 class="hero-title">Shop<span>Mind</span></h1>'
        f'<p class="hero-sub">Instant, accurate answers for <b>{html.escape(shop_name)}</b> customers. '
        'The owner steps in only when it matters.</p>'
        '<div class="chips">'
        f'<span class="chip live">● {html.escape(source_label)}</span>'
        '<span class="chip">🔎 Lookup agent</span><span class="chip">✍️ Reply agent</span>'
        '<span class="chip">🛡️ Review agent</span></div></div>'
    )


def stats_html(items) -> str:
    cards = "".join(
        f'<div class="stat"><div class="num">{html.escape(str(v))}</div><div class="lbl">{html.escape(icon)} {html.escape(label)}</div></div>'
        for icon, v, label in items
    )
    return f'<div class="stats">{cards}</div>'


def steps_html(needs_owner: bool) -> str:
    verdict = (
        '<span class="verdict owner">👤 Sent to owner</span>' if needs_owner
        else '<span class="verdict ok">⚡ Auto-sent</span>'
    )
    return (
        '<div class="steps"><span class="step">🔎 Lookup</span><span class="arrow">→</span>'
        '<span class="step">✍️ Reply</span><span class="arrow">→</span>'
        f'<span class="step">🛡️ Review</span><span class="arrow">→</span>{verdict}</div>'
    )
