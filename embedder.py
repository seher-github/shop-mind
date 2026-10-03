"""Tiny local text embedding: hashed character n-grams.

No model download, no API key, works offline. Because it looks at 3-letter pieces of words,
it still matches spelling variations and typos ("jacket" ~ "jakcet", "kurta" ~ "kurtaa").
It is a "fuzzy" vector search, not a deep-learning one. To upgrade later, replace embed()
with a call to a real embedding model (for example Gemini) - nothing else needs to change.
"""
import math
import re
import zlib

DIM = 384

FILLER = {
    "the", "a", "an", "is", "are", "do", "does", "you", "your", "have", "has", "what", "how", "can", "i", "me",
    "my", "we", "to", "of", "in", "on", "for", "and", "or", "it", "this", "that", "with", "please",
    "hai", "hain", "ka", "ki", "ke", "kya", "aap", "ap", "mein", "me", "se", "ko", "ye", "yeh", "woh", "wo", "bhai",
}


def _features(text: str) -> list:
    words = [w for w in re.findall(r"[a-z0-9\u0600-\u06ff]+", str(text).lower()) if w not in FILLER]
    feats = []
    for w in words:
        padded = f"<{w}>"
        feats += [padded[i:i + 3] for i in range(max(1, len(padded) - 2))]
        feats.append("w_" + w)
    return feats


def embed(text: str) -> list:
    vec = [0.0] * DIM
    for f in _features(text):
        h = zlib.crc32(f.encode("utf-8"))
        vec[h % DIM] += 1.0 if (h >> 16) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
