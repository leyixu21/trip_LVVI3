"""NER-based PII redaction (see scripts/multimodal_extraction/apply_privacy_ner.py
for the original one-off version this replaces for the revision-1 pipeline).

Two changes from that version, both needed to run over the full
fused-chunks dataset rather than a single exploratory slice:
- Batched inference: redact_texts() flattens every window from every input
  text into one list and makes a single pipeline() call with batch_size,
  instead of calling the HF pipeline one sentence at a time.
- Offset-based redaction: uses each entity's character start/end
  (aggregation_strategy="simple" gives exact offsets) to splice the
  original string, instead of `sent.replace(entity['word'], ...)`, which
  replaces every occurrence of that substring, not just the tagged span.
"""
from __future__ import annotations

import re

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# Common emoji/pictograph/symbol blocks, plus the variation-selector and
# zero-width-joiner codepoints emoji sequences are built from.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "\uFE0F"  # variation selector-16 (forces emoji presentation)
    "\u200D"  # zero-width joiner (chains compound emoji, e.g. family/flag sequences)
    "]+"
)


def scrub_structured_pii(texts: list[str], min_phone_digits: int, min_digit_run: int) -> list[str]:
    """Regex pass for PII patterns NER doesn't reliably catch: emails and
    URLs (contact info), HTML tags and emojis (formatting noise), and
    phone-number-like or otherwise long digit strings. Emails/URLs/numbers
    become bracket placeholders, consistent with entity redaction; tags and
    emojis carry no information so they're dropped outright rather than
    replaced.

    min_phone_digits/min_digit_run keep short numbers that are normal in
    these transcripts (distances, times, counts -- e.g. "15 km", "3.5
    hours") from being redacted: only digit runs at or above min_digit_run,
    or digit-plus-separator sequences with at least min_phone_digits digit
    characters, are treated as PII."""
    phone_pattern = re.compile(r"(?<!\w)\+?(?:\d[\s\-.]?){%d,}\d(?!\w)" % (min_phone_digits - 1))
    digit_run_pattern = re.compile(r"(?<!\w)\d{%d,}(?!\w)" % min_digit_run)

    def _scrub_one(text: str) -> str:
        text = _EMAIL_PATTERN.sub("[EMAIL]", text)
        text = _URL_PATTERN.sub("[URL]", text)
        text = _HTML_TAG_PATTERN.sub("", text)
        text = _EMOJI_PATTERN.sub("", text)
        text = phone_pattern.sub("[NUMBER]", text)
        text = digit_run_pattern.sub("[NUMBER]", text)
        return text

    return [_scrub_one(t) for t in texts]


def split_into_windows(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split text into (offset, piece) windows of at most max_chars each,
    breaking on whitespace so a word -- and therefore a potential entity --
    is never cut in half. Needed because the NER model's max position
    embeddings (512 tokens) is far shorter than some real fused_text values
    (see config.PRIVACY_NER_WINDOW_MAX_CHARS's comment)."""
    windows = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + max_chars, n)
        if end < n:
            ws = text.rfind(" ", pos, end)
            if ws > pos:
                end = ws
        if end > pos:
            windows.append((pos, text[pos:end]))
        pos = end
        while pos < n and text[pos] == " ":
            pos += 1
    return windows


def redact_texts(
    texts: list[str], ner_pipeline, entity_types: list[str], score_threshold: float,
    window_max_chars: int, batch_size: int,
) -> list[str]:
    """Redacts PII entities (entity_group in entity_types, above
    score_threshold) from every string in texts, returning a same-length,
    same-order list of redacted strings. Each matched span is replaced with
    "[{entity_group}]" (e.g. "[PER]") in place, leaving everything else --
    including entity types not in entity_types -- untouched.

    Every window from every input text is flattened into one list and sent
    through a single batched pipeline() call, so throughput scales with
    batch_size/GPU headroom instead of the number of input texts."""
    # windows_by_text[i] = list of (offset, window_text) for texts[i]
    windows_by_text = [split_into_windows(t, window_max_chars) for t in texts]

    flat_windows = []  # (text_index, offset, window_text)
    for i, windows in enumerate(windows_by_text):
        for offset, window_text in windows:
            flat_windows.append((i, offset, window_text))

    if not flat_windows:
        return list(texts)

    flat_texts = [w[2] for w in flat_windows]
    entity_types = set(entity_types)
    all_entities = ner_pipeline(flat_texts, batch_size=batch_size)
    # pipeline() returns a flat list of per-item entity lists when given a
    # list input, one-to-one with flat_texts.

    # spans_by_text[i] = list of (global_start, global_end, entity_group)
    spans_by_text: list[list[tuple[int, int, str]]] = [[] for _ in texts]
    for (text_index, offset, _window_text), entities in zip(flat_windows, all_entities):
        for entity in entities:
            if entity["entity_group"] not in entity_types or entity["score"] <= score_threshold:
                continue
            spans_by_text[text_index].append((offset + entity["start"], offset + entity["end"], entity["entity_group"]))

    redacted = []
    for text, spans in zip(texts, spans_by_text):
        if not spans:
            redacted.append(text)
            continue
        spans.sort(key=lambda s: s[0])
        pieces = []
        cursor = 0
        for start, end, entity_group in spans:
            if start < cursor:
                continue  # overlapping span from a window boundary artifact -- keep the first
            pieces.append(text[cursor:start])
            pieces.append(f"[{entity_group}]")
            cursor = end
        pieces.append(text[cursor:])
        redacted.append("".join(pieces))
    return redacted
