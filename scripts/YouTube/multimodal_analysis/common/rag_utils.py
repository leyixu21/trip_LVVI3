"""Lightweight RAG retriever over taxonomy definitions.

Deliberately not FAISS: the taxonomy has at most a few hundred rows, so a
dense numpy matmul is both simpler to ship on CSC nodes and fast enough.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TaxonomyIndex:
    taxonomy_name: str
    labels: list[str]
    texts: list[str]  # retrieval/embedding text (short: label + definition)
    embeddings: np.ndarray  # [N, D], L2-normalized
    display_texts: list[str] | None = None  # full codebook text shown in prompts; None -> use texts

    def prompt_text(self, i: int) -> str:
        """The text to show an LLM for label i -- the full codebook entry when
        the index carries one, else the retrieval text."""
        if self.display_texts is not None:
            return self.display_texts[i]
        return self.texts[i]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TaxonomyIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


class TaxonomyRetriever:
    def __init__(self, index: TaxonomyIndex, embed_fn):
        self.index = index
        self.embed_fn = embed_fn

    def retrieve(self, query_text: str, top_k: int) -> list[tuple[str, str, float]]:
        """Returns [(label, display_text, score), ...] sorted by score desc."""
        query_emb = self.embed_fn([query_text])[0]
        scores = self.index.embeddings @ query_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.index.labels[i], self.index.prompt_text(i), float(scores[i])) for i in top_idx]


def build_taxonomy_index(taxonomy_name: str, labels: list[str], texts: list[str], embed_fn,
                         display_texts: list[str] | None = None) -> TaxonomyIndex:
    embeddings = embed_fn(texts)
    return TaxonomyIndex(
        taxonomy_name=taxonomy_name, labels=labels, texts=texts,
        embeddings=embeddings, display_texts=display_texts,
    )
