from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import time


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    sentences: list
    score: float


class LocalBM25:
    version = "rewrite-bm25-v1_k1=1.2_b=0.75_unicode-word"

    def __init__(self, corpus, snapshot_hash=None):
        path = Path(corpus)
        self.snapshot_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if snapshot_hash and self.snapshot_hash != snapshot_hash:
            raise ValueError("corpus_snapshot_hash_mismatch")
        self.passages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.counts = [Counter(self.tokens(p["title"] + " " + " ".join(s["text"] for s in p["sentences"]))) for p in self.passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(term for c in self.counts for term in c)

    @staticmethod
    def tokens(text):
        return re.findall(r"\w+", text.casefold())

    def search(self, query, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        n = len(self.passages)
        scores = []
        for idx, counts in enumerate(self.counts):
            score = 0
            for term in set(self.tokens(query)):
                tf = counts[term]
                if tf:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.lengths[idx] / max(1, self.average_length)))
            scores.append((score, self.passages[idx]))
        scores.sort(key=lambda p: (-p[0], p[1]["id"]))
        return [Passage(p["id"], p["title"], deepcopy(p["sentences"]), score) for score, p in scores[:top_k]]

    def search_record(self, query, top_k=5):
        start = time.monotonic()
        results = self.search(query, top_k)
        return {"query": query, "top_k": top_k, "retriever_version": self.version,
                "snapshot_hash": self.snapshot_hash, "search_calls": 1,
                "duration_monotonic": time.monotonic() - start, "passages": [asdict(p) for p in results]}
