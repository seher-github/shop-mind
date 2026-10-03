"""Hybrid retriever: keyword search + vector search (ChromaDB), plus owner-correction lookup.

One HybridIndex is built per shop (demo or uploaded data) and kept in the visitor's session.
"""
import difflib
import uuid

import chromadb

from embedder import embed
from tools.shop_data_tool import (
    GENERIC_WORDS, _format_product, _policy_keys, _price_text, _product_tokens, _split_label, _tokens,
)

VEC_MIN = 0.30          # minimum cosine similarity for a vector-only hit
CORR_MIN = 0.45         # minimum similarity for an owner-correction example
_REGISTRY = []          # names of live collections (old ones are dropped to save memory)
MAX_LIVE_INDEXES = 20


def _product_doc(p: dict) -> str:
    colors = " ".join(c for c in p.get("colors", []) if c)
    return f"{p['name']} {p.get('category', '')} {colors} {p.get('notes', '')}"


class HybridIndex:
    def __init__(self, shop: dict, corrections=None):
        self.shop = shop
        self.corrections = []
        self._client = chromadb.EphemeralClient()
        self.name = "sm_" + uuid.uuid4().hex[:12]
        self._col = self._client.create_collection(
            name=self.name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        _REGISTRY.append(self.name)
        while len(_REGISTRY) > MAX_LIVE_INDEXES:
            old = _REGISTRY.pop(0)
            try:
                self._client.delete_collection(old)
            except Exception:  # noqa: BLE001
                pass

        self._counts = {"product": 0, "policy": 0, "correction": 0}
        self._policy_items = list(shop["policies"].items())
        self._add_batch("product", [_product_doc(p) for p in shop["products"]], "p")
        self._add_batch("policy", [f"{label} {text}" for label, text in self._policy_items], "l")
        for c in corrections or []:
            self.add_correction(c["question"], c["reply"])

        vocab = set()
        for p in shop["products"]:
            vocab |= _product_tokens(p)
        for label, _ in self._policy_items:
            vocab |= _tokens(label)
        self._vocab = sorted(vocab)

    # ---------- building ----------
    def _add_batch(self, kind: str, docs: list, prefix: str):
        for start in range(0, len(docs), 500):
            chunk = docs[start:start + 500]
            if not chunk:
                continue
            self._col.add(
                ids=[f"{prefix}{start + i}" for i in range(len(chunk))],
                embeddings=[embed(d) for d in chunk],
                documents=chunk,
                metadatas=[{"type": kind, "idx": start + i} for i in range(len(chunk))],
            )
        self._counts[kind] += len(docs)

    def add_correction(self, question: str, reply: str):
        n = len(self.corrections)
        self.corrections.append({"question": question, "reply": reply})
        self._col.add(
            ids=[f"c{n}"], embeddings=[embed(question)], documents=[question],
            metadatas=[{"type": "correction", "idx": n}],
        )
        self._counts["correction"] += 1

    def close(self):
        try:
            self._client.delete_collection(self.name)
            if self.name in _REGISTRY:
                _REGISTRY.remove(self.name)
        except Exception:  # noqa: BLE001
            pass

    # ---------- searching ----------
    def _expand(self, tokens: set) -> set:
        """Typo tolerance: 'jakcet' -> also search 'jacket'."""
        out = set(tokens)
        for t in tokens:
            if len(t) >= 4 and t not in self._vocab and t not in GENERIC_WORDS:
                close = difflib.get_close_matches(t, self._vocab, n=1, cutoff=0.8)
                out.update(close)
        return out

    def _vector(self, kind: str, queries: list, k: int) -> list:
        """Returns [(idx, similarity)] best first, for one document type."""
        n = self._counts[kind]
        if n == 0:
            return []
        try:
            res = self._col.query(
                query_embeddings=[embed(q) for q in queries if q.strip()],
                n_results=min(k, n), where={"type": kind},
            )
        except Exception:  # noqa: BLE001  (vector part is optional; keyword search still works)
            return []
        best = {}
        for metas, dists in zip(res["metadatas"], res["distances"]):
            for m, d in zip(metas, dists):
                sim = 1.0 - d
                best[m["idx"]] = max(best.get(m["idx"], -1.0), sim)
        return sorted(best.items(), key=lambda x: -x[1])

    def search(self, query: str, original: str = "") -> dict:
        shop = self.shop
        currency = shop.get("currency", "")
        products, policies = shop["products"], self._policy_items
        queries = [q for q in {query, original} if q and q.strip()]
        tokens = self._expand(_tokens(f"{query} {original}"))

        # ----- catalogue overview ("what do you sell?") -----
        base_tokens = _tokens(query or original)
        if not base_tokens or all(t in GENERIC_WORDS or t.rstrip("s") in GENERIC_WORDS for t in base_tokens):
            lines = [f"- {p['name']} ({p.get('category') or 'n/a'}): {_price_text(p, currency)}" for p in products[:15]]
            return self._pack([], [], [], "CATALOGUE OVERVIEW:\n" + "\n".join(lines), "high", overview=True)

        # ----- products: keyword + vector, fused -----
        kw = sorted(((len(tokens & _product_tokens(p)), i) for i, p in enumerate(products)), reverse=True)
        kw_rank = [i for score, i in kw if score > 0][:5]
        vec = self._vector("product", queries, 5)
        vec_hits = {i: s for i, s in vec if s >= VEC_MIN}
        fused = {}
        for r, i in enumerate(kw_rank):
            fused[i] = fused.get(i, 0) + 1 / (60 + r)
        for r, (i, s) in enumerate(vec):
            if i in kw_rank or s >= VEC_MIN:
                fused[i] = fused.get(i, 0) + 1 / (60 + r)
        prod_hits = []
        for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:3]:
            via = "keyword + semantic" if (i in kw_rank and i in vec_hits) else ("keyword" if i in kw_rank else "semantic")
            prod_hits.append((i, via, vec_hits.get(i)))

        # ----- policies: keyword (scope-aware) first, vector only as backup -----
        groups = {}
        for label, text in policies:
            base, scope = _split_label(label)
            if tokens & _policy_keys(base):
                groups.setdefault(base, []).append((label, scope, text))
        pol_hits = []
        for rows in groups.values():
            scored = [(len(tokens & _tokens(scope)) if scope else 0, label, text) for label, scope, text in rows]
            best = max(s[0] for s in scored)
            for _, label, text in [s for s in scored if s[0] == best][:5]:
                pol_hits.append((label, text, "keyword", None))
        if not pol_hits:
            for i, s in self._vector("policy", queries, 3):
                if s >= VEC_MIN and len(pol_hits) < 2:
                    pol_hits.append((policies[i][0], policies[i][1], "semantic", s))

        # ----- owner corrections (style examples) -----
        corr_hits = []
        if self._counts["correction"]:
            for i, s in self._vector("correction", queries, 2):
                if s >= CORR_MIN:
                    c = self.corrections[i]
                    corr_hits.append({"question": c["question"], "reply": c["reply"], "similarity": round(s, 2)})

        # ----- confidence -----
        keyword_found = any(v.startswith("keyword") for _, v, _ in prod_hits) or any(v == "keyword" for *_, v, _ in pol_hits)
        sims = [s for *_, s in prod_hits if s] + [h[3] for h in pol_hits if h[3]]
        if not prod_hits and not pol_hits:
            confidence = "none"
        elif keyword_found:
            confidence = "high"
        else:
            confidence = "medium" if sims and max(sims) >= 0.5 else "low"

        # ----- text for the agents -----
        parts = []
        if prod_hits:
            parts.append("PRODUCTS:\n" + "\n".join(_format_product(products[i], currency) for i, _, _ in prod_hits))
        if pol_hits:
            parts.append("POLICIES:\n" + "\n".join(f"POLICY ({label}): {text}" for label, text, _, _ in pol_hits))
        if not parts:
            catalog = ", ".join(p["name"] for p in products[:20])
            facts = (f"NO_MATCH: nothing in the shop data matches this question. The shop sells: {catalog}. "
                     f"Policy topics available: {', '.join(l for l, _ in policies) or 'none'}.")
        else:
            facts = "\n".join(parts)
        return self._pack(prod_hits, pol_hits, corr_hits, facts, confidence)

    def _pack(self, prod_hits, pol_hits, corr_hits, facts, confidence, overview=False) -> dict:
        examples = "\n\n".join(
            f"Customer: {c['question'][:200]}\nOwner's reply: {c['reply'][:300]}" for c in corr_hits
        )
        return {
            "facts_text": facts,
            "confidence": confidence,
            "overview": overview,
            "products": [{"name": self.shop["products"][i]["name"], "via": via, "similarity": s} for i, via, s in prod_hits],
            "policies": [{"label": l, "via": v, "similarity": s} for l, _, v, s in pol_hits],
            "corrections": corr_hits,
            "examples_text": examples,
        }
