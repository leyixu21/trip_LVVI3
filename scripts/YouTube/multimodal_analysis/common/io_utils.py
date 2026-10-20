"""Taxonomy loading and safe parsing of LLM label output.

The legacy pipeline parsed model output with `eval(response)`, which is both
a code-injection risk and silently drops a whole chunk's labels on any
formatting hiccup. This module extracts a JSON list robustly and validates
every label against the allowed taxonomy instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd

_LIST_PATTERN = re.compile(r"\[.*?\]", re.DOTALL)


def parse_label_list(response_text: str, allowed_labels: list[str]) -> list[str]:
    """Extract a list of labels from raw LLM text and keep only labels that
    (case-insensitively) match the allowed taxonomy. Never raises."""
    if not response_text:
        return []

    match = _LIST_PATTERN.search(response_text)
    if not match:
        return []

    raw = match.group(0)
    candidates = None
    try:
        candidates = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # tolerate single-quoted python-literal style lists
            candidates = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            candidates = None

    if not isinstance(candidates, list):
        return []

    allowed_lookup = {label.strip().lower(): label for label in allowed_labels}
    result = []
    for c in candidates:
        if not isinstance(c, str):
            continue
        key = c.strip().lower()
        if key in allowed_lookup:
            label = allowed_lookup[key]
            if label not in result:
                result.append(label)
    return result


@dataclass
class Taxonomy:
    name: str
    label_col: str
    group_col: str
    definition_col: str | None
    example_col: str | None
    include_when_col: str | None = None
    exclude_when_col: str | None = None
    distinguish_col: str | None = None
    df: pd.DataFrame | None = None

    @property
    def labels(self) -> list[str]:
        return self.df[self.label_col].dropna().unique().tolist()

    def _row(self, label: str):
        row = self.df[self.df[self.label_col] == label]
        return None if row.empty else row.iloc[0]

    def definition_text(self, label: str) -> str:
        """Short label+definition (+examples) string. This is the RETRIEVAL
        text embedded into the RAG index -- kept short on purpose (the
        embedding model truncates ~128 tokens, and negated criteria must not
        become positive retrieval signal)."""
        row = self._row(label)
        if row is None:
            return ""
        parts = [f"{self.label_col.replace('_', ' ').title()}: {label}"]
        if self.definition_col and pd.notna(row.get(self.definition_col)):
            parts.append(f"Definition: {row[self.definition_col]}")
        if self.example_col and pd.notna(row.get(self.example_col)):
            parts.append(f"Examples: {row[self.example_col]}")
        return " | ".join(parts)

    def display_text(self, label: str) -> str:
        """Full codebook entry shown to the LLM in classification and
        adjudication prompts: definition_text() plus, when the taxonomy CSV
        provides them, the codebook's include-when / do-not-include-when /
        distinguish-from criteria."""
        text = self.definition_text(label)
        row = self._row(label)
        if row is None or not text:
            return text
        for col, heading in (
            (self.include_when_col, "Include when"),
            (self.exclude_when_col, "Do not include when"),
            (self.distinguish_col, "Distinguish from"),
        ):
            if col and pd.notna(row.get(col)) and row[col]:
                text += f" | {heading}: {row[col]}"
        return text

    def corpus(self) -> list[tuple[str, str]]:
        """Returns [(label, retrievable_text), ...] for RAG indexing."""
        return [(label, self.definition_text(label)) for label in self.labels]

    def display_corpus(self) -> list[tuple[str, str]]:
        """[(label, display_text), ...] in the same order as corpus()."""
        return [(label, self.display_text(label)) for label in self.labels]


def load_taxonomy(name: str, cfg_entry: dict) -> Taxonomy:
    df = pd.read_csv(cfg_entry["path"])
    return Taxonomy(
        name=name,
        label_col=cfg_entry["label_col"],
        group_col=cfg_entry["group_col"],
        definition_col=cfg_entry.get("definition_col"),
        example_col=cfg_entry.get("example_col"),
        include_when_col=cfg_entry.get("include_when_col"),
        exclude_when_col=cfg_entry.get("exclude_when_col"),
        distinguish_col=cfg_entry.get("distinguish_col"),
        df=df,
    )
