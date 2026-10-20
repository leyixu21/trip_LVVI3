# trip_LVVI3

Comparing nature-based tourism & outdoor recreation activities across three data sources —
**LVVI3** survey, **YouTube** videos, and **Flickr** photos — in Finnish state-owned lands.

- `scripts/` — all code; see **`scripts/README.md`** for the numbered pipeline
  (per source: `data_preprocessing/` → `multimodal_analysis/` → `RQs_analysis/`).
- `data/` — all data (git-ignored). Preprocessing outputs are **CSV** (pkl converted 2026-10-20).
- `data/_archive/old_llm_subactivity/` — old Qwen3-VL sub_activity outputs, kept until the
  clef-model labels are aggregated (see its README).
- Facial recognition was removed from the project in the 2026-10 reorganization.
- `AGENTS.md` — rules for AI agents working in this repo (token-saving, never-read list).
