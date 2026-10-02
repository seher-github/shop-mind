"""Shop data search + a CrewAI tool built for ONE shop.

The search uses no AI, so it costs nothing and never invents facts.
The tool is built per request, so one visitor's uploaded data is never visible to another.
"""
import json
import re
from pathlib import Path

from crewai.tools import tool

DEMO_FILE = Path(__file__).resolve().parent.parent / "data" / "shop.json"

STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "you", "your", "have", "has",
    "in", "on", "of", "for", "to", "and", "or", "i", "me", "my", "we", "can",
    "what", "how", "much", "many", "any", "there", "it", "this", "that", "with",
    "size", "sizes", "please", "hi", "hello", "want", "need", "get", "buy",
    "policy", "policies", "tell", "about", "show", "which", "when", "where",
}

# Words that just mean "show me what you sell".
GENERIC_WORDS = {
    "product", "catalog", "catalogue", "collection", "sell", "selling", "range",
    "stock", "available", "availability", "item", "clothes", "clothing", "wear", "price", "cost",
}

# Words that point to common policy topics (used when a topic name matches one of these).
SYNONYMS = {
    "delivery": {"deliver", "delivery", "shipping", "ship", "shipped", "arrive", "courier", "day", "express", "international"},
        "return": {"return", "refund", "exchange", "swap"},
    "exchange": {"exchange", "swap", "change", "different", "replace"},
    "payment": {"pay", "payment", "cash", "card", "cod", "credit", "debit"},
    "hour": {"open", "close", "closed", "hour", "time", "sunday", "saturday", "monday"},
}


def normalize_shop(shop: dict) -> dict:
    """Make sure every product uses the stock_by_color layout."""
    for p in shop["products"]:
        if "stock_by_color" not in p:
            p["stock_by_color"] = {"": p.get("stock", {})}
        p.setdefault("notes", "")
        p.setdefault("category", "")
    return shop


def load_demo_shop() -> dict:
    return normalize_shop(json.loads(DEMO_FILE.read_text(encoding="utf-8")))


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    cleaned = set()
    for w in words:
        if w in STOPWORDS:
            continue
        # crude plural handling: "hoodies" -> "hoodie", "shirts" -> "shirt"
        if w.endswith("ies") and len(w) > 4:
            cleaned.add(w[:-3] + "y")
            cleaned.add(w[:-1])
        elif w.endswith("s") and len(w) > 3:
            cleaned.add(w[:-1])
        cleaned.add(w)
    return cleaned


def _product_tokens(p: dict) -> set:
    colors = [c for c in p.get("colors", []) if c]
    return _tokens(" ".join([p["name"], p.get("category", ""), " ".join(colors)]))


def _qty_text(q: int) -> str:
    if q == 0:
        return "OUT OF STOCK"
    if q < 0:
        return "available (quantity not listed)"
    if q <= 3:
        return f"only {q} left"
    return f"in stock ({q})"


def _price_text(p: dict, currency: str) -> str:
    return f"{p['price']:.2f} {currency}" if p.get("price") is not None else "price not listed"


def _format_product(p: dict, currency: str) -> str:
    stock_parts = []
    for color, sizes in p["stock_by_color"].items():
        label = color or "all colours"
        sizes_txt = "; ".join(f"{s}: {_qty_text(q)}" for s, q in sizes.items())
        stock_parts.append(f"{label} -> {sizes_txt}")
    colors = ", ".join(c for c in p.get("colors", []) if c) or "not listed"
    return (
        f"PRODUCT: {p['name']} | category: {p.get('category') or 'n/a'} | price: {_price_text(p, currency)} | "
        f"colours: {colors} | stock -> {' || '.join(stock_parts) or 'not listed'} | notes: {p.get('notes') or 'none'}"
    )


def _policy_keys(topic: str) -> set:
    keys = _tokens(topic)
    for canon, syn in SYNONYMS.items():
        if keys & (syn | _tokens(canon)):
            keys |= syn
    return keys


def _split_label(label: str):
    """'delivery (Islamabad)' -> ('delivery', 'Islamabad'); 'hours' -> ('hours', '')."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
    return (m.group(1), m.group(2)) if m else (label, "")


def search_shop_data(query: str, shop: dict) -> str:
    tokens = _tokens(query)
    currency = shop.get("currency", "")
    products = shop["products"]
    policies = shop.get("policies", {})
    results = []

    # "What do you sell?" style questions -> short catalogue overview.
    if not tokens or all(t in GENERIC_WORDS or t.rstrip("s") in GENERIC_WORDS for t in tokens):
        lines = [f"- {p['name']} ({p.get('category') or 'n/a'}): {_price_text(p, currency)}" for p in products[:15]]
        extra = f" (showing 15 of {len(products)})" if len(products) > 15 else ""
        return f"CATALOGUE OVERVIEW{extra}:\n" + "\n".join(lines)

    scored = [(len(tokens & _product_tokens(p)), p) for p in products]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    for _, p in scored[:3]:
        results.append(_format_product(p, currency))

    # Policies are grouped by topic ("delivery", "returns"...). Inside a matching topic we keep the rows
    # whose scope the customer mentioned ("delivery to Islamabad" -> only the Islamabad row).
    # If no scope was mentioned, every row of that topic is kept.
    groups = {}
    for label, text in policies.items():
        base, scope = _split_label(label)
        if tokens & _policy_keys(base):
            groups.setdefault(base, []).append((label, scope, text))
    for rows in groups.values():
        scored = [(len(tokens & _tokens(scope)) if scope else 0, label, text) for label, scope, text in rows]
        best = max(s[0] for s in scored)
        for _, label, text in [s for s in scored if s[0] == best][:5]:
            results.append(f"POLICY ({label}): {text}")

    # Fallback: look inside the policy texts.
    if not results and policies:
        ranked = sorted(((len(tokens & _tokens(t)), k, t) for k, t in policies.items()), reverse=True)
        for score, topic, text in ranked[:2]:
            if score > 0:
                results.append(f"POLICY ({topic}): {text}")

    if not results:
        catalog = ", ".join(f"{p['name']}" for p in products[:20])
        return (
            "NO_MATCH: nothing in the shop data matches this search. "
            f"The shop sells: {catalog}. "
            f"Policy topics available: {', '.join(policies) or 'none'}."
        )
    return "\n".join(results)





def build_lookup_tool(shop: dict):
    """Create the CrewAI tool bound to this one shop's data."""

    @tool("Shop Data Lookup")
    def shop_data_lookup(query: str) -> str:
        """Search the shop's real product data (price, colours, stock by size) and
        policies (delivery, returns, exchange, payment, opening hours).
        Input: a short search phrase such as 'blue denim jacket' or 'return policy'."""
        return search_shop_data(query, shop)

    return shop_data_lookup
