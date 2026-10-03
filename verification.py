"""Answer verification (plain code, no AI): checks the draft reply against the retrieved facts.

It looks for numbers (prices, days, quantities), clothing sizes and risky promises
(free delivery, refund, discount...) that appear in the draft but NOT in the retrieved data.
"""
import re

NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
SIZE_RE = re.compile(r"\b(XXXL|XXL|XL|XS|S|M|L)\b")
RISKY_PHRASES = [
    "free delivery", "free shipping", "cash on delivery", "refund", "store credit", "exchange", "discount",
    "warranty", "guarantee", "same day", "same-day", "express", "international", "installment", "money back",
]


def _numbers(text: str) -> set:
    out = set()
    for m in NUM_RE.findall(str(text)):
        s = m.replace(",", "").rstrip(".")
        try:
            out.add(round(float(s), 2))
        except ValueError:
            pass
    return out


def verify(draft: str, docs_text: str, message: str = "") -> dict:
    """Returns {'ok': bool, 'unsupported': [..], 'checked': int}."""
    allowed_numbers = _numbers(docs_text) | _numbers(message)
    draft_numbers = _numbers(draft)
    unsupported = []

    bad_numbers = sorted(n for n in draft_numbers if n not in allowed_numbers)
    for n in bad_numbers:
        unsupported.append(f"number {n:g} is not in the shop data")

    docs_sizes = set(SIZE_RE.findall(docs_text)) | set(SIZE_RE.findall(message))
    for s in sorted(set(SIZE_RE.findall(draft)) - docs_sizes):
        unsupported.append(f"size {s} is not in the shop data")

    docs_l, msg_l, draft_l = docs_text.lower(), message.lower(), draft.lower()
    for phrase in RISKY_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", draft_l) and phrase not in docs_l and phrase not in msg_l:
            unsupported.append(f"'{phrase}' is promised but not in the shop data")

    checked = len(draft_numbers) + len(set(SIZE_RE.findall(draft)))
    return {"ok": not unsupported, "unsupported": unsupported, "checked": checked}
