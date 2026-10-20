"""
Local, GPU-free sanity checks for the pure-Python logic in the LVVI3
multimodal pipeline: label parsing, chunk fusion/alignment, RAG retrieval,
privacy scrubbing, and video-level aggregation. Model-loading stages (shot
detection on real video, VLM captioning, Whisper, the actual LLM calls)
can't run here without CSC's GPUs/Allas data, but everything downstream of
"some model produced these numbers/strings" is verified against synthetic
inputs.

Run with the project's cpu-env python (has numpy/pandas/torch/transformers;
the login shell's bare python3 may lack torch/transformers):
  /projappl/project_2006782/leyixu/cpu-env/bin/python3 tests/test_pipeline_logic.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, io_utils, rag_utils
from common.privacy_ner import redact_texts, scrub_structured_pii, split_into_windows
from common.shot_detection import build_tagged_segments, expand_long_segments, group_by_shot, subsample_shots
from common.text_chunking import FrameCaption, TranscriptSegment, build_fused_chunks
from common.whisper_utils import audio_event_summary, filter_hallucinated_segments, is_hallucinated_segment

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# ------------------------------------------------------------------
# config: activities-only taxonomy yields the 22 unique LVVI3 labels
# ------------------------------------------------------------------
def test_taxonomy_config():
    print("\n[config: LVVI3 activities taxonomy]")
    check("exactly one taxonomy (activities)", list(config.TAXONOMIES) == ["activities"], list(config.TAXONOMIES))
    check("RAG_TOP_K covers all labels", config.RAG_TOP_K >= 22, config.RAG_TOP_K)
    check("vote threshold is 1 (single-model ensemble)", config.ENSEMBLE_VOTE_MIN_SELECTED == 1)
    check("transcripts input is data/YouTube/transcripts.jsonl",
          config.TRANSCRIPTS_JSONL.name == "transcripts.jsonl", config.TRANSCRIPTS_JSONL)

    taxonomy = io_utils.load_taxonomy("activities", config.TAXONOMIES["activities"])
    csv_labels = pd.read_csv(config.TAXONOMIES["activities"]["path"])["activity"].dropna().unique().tolist()
    check("Taxonomy.labels has 22 unique labels", len(taxonomy.labels) == 22, len(taxonomy.labels))
    check("labels match the CSV's unique activity values, deduped",
          sorted(taxonomy.labels) == sorted(csv_labels), taxonomy.labels[:5])
    check("labels are unique (no sub-activity duplicates)",
          len(taxonomy.labels) == len(set(taxonomy.labels)))
    check("retrieval text is just the label prefixed by its column name (no definition cols)",
          all(taxonomy.definition_text(l) == f"Activity: {l}" for l in taxonomy.labels))


# ------------------------------------------------------------------
# io_utils.parse_label_list
# ------------------------------------------------------------------
def test_parse_label_list():
    print("\n[parse_label_list]")
    allowed = ["Skiing", "Kayaking", "Bird watching"]

    check("clean json list", io_utils.parse_label_list('["Skiing", "Kayaking"]', allowed) == ["Skiing", "Kayaking"])
    check("python-literal single quotes", io_utils.parse_label_list("['Skiing']", allowed) == ["Skiing"])
    check(
        "surrounded by prose",
        io_utils.parse_label_list('Sure, here is the answer: ["Skiing"] hope that helps', allowed) == ["Skiing"],
    )
    check("case-insensitive match", io_utils.parse_label_list('["skiing"]', allowed) == ["Skiing"])
    check("drops hallucinated label", io_utils.parse_label_list('["Skiing", "Snowboarding"]', allowed) == ["Skiing"])
    check("empty list", io_utils.parse_label_list("[]", allowed) == [])
    check("garbage text -> []", io_utils.parse_label_list("I cannot help with that.", allowed) == [])
    check("empty string -> []", io_utils.parse_label_list("", allowed) == [])
    check("dedupes repeated labels", io_utils.parse_label_list('["Skiing", "Skiing"]', allowed) == ["Skiing"])
    check(
        "no eval() code execution risk",
        io_utils.parse_label_list('["Skiing"]; __import__("os").system("echo pwned")', allowed) == ["Skiing"],
    )


# ------------------------------------------------------------------
# text_chunking: window merge + caption assignment + CLIP-sim dedup
# ------------------------------------------------------------------
_FAKE_EMBED_DIM = 256


def fake_embed_fn(texts):
    """Deterministic fixed-width hashed bag-of-words embedding so dedup/
    retrieval logic is testable without downloading CLIP or sentence-
    transformers. Unlike a per-call vocabulary, hashing keeps the output
    dimension constant across independent calls (as any real embedding
    model would), so index and query embeddings stay comparable."""
    mat = np.zeros((len(texts), _FAKE_EMBED_DIM))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            mat[i, hash(w) % _FAKE_EMBED_DIM] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def test_chunk_fusion():
    print("\n[text_chunking.build_fused_chunks]")
    segments = [
        TranscriptSegment(start=0.0, end=5.0, text="We started our hike early in the morning."),
        TranscriptSegment(start=6.0, end=12.0, text="The trail was covered in fresh snow."),
        TranscriptSegment(start=25.0, end=30.0, text="We stopped by the lake for lunch."),
    ]
    frame_captions = [
        FrameCaption(start=0.5, end=2.0, shot_index=0, frame_names=["f1.jpg"], caption="a person walking on a snowy trail"),
        FrameCaption(start=2.0, end=3.5, shot_index=1, frame_names=["f2.jpg"], caption="a person walking on a snowy trail"),  # near-dup
        FrameCaption(start=7.0, end=9.0, shot_index=2, frame_names=["f3.jpg"], caption="a forest covered in snow"),
        FrameCaption(start=26.0, end=28.0, shot_index=3, frame_names=["f4.jpg"], caption="a lake surrounded by mountains"),
        # A shot that would have straddled a fixed 20s cutoff (15-25s) --
        # shot-aware windowing must instead cut AT this shot's end (25.0),
        # keeping the whole shot in one chunk rather than splitting it.
        FrameCaption(start=15.0, end=25.0, shot_index=4, frame_names=["f5.jpg", "f6.jpg"], caption="a wide panning shot of the valley"),
    ]

    chunks = build_fused_chunks(
        segments=segments, frame_captions=frame_captions, video_id="vid1",
        window_seconds=20.0, embed_fn=fake_embed_fn, dedup_threshold=0.95,
    )

    check("produces 2 non-empty windows (0.5-25s, 25-28s)", len(chunks) == 2, str(chunks))
    win0 = next(c for c in chunks if c.chunk_index == 0)
    win1 = next(c for c in chunks if c.chunk_index == 1)

    check("window 0 cuts exactly at the panning shot's end (25.0s)", win0.end == 25.0, win0)
    check(
        "window 0 merges both early transcript segments",
        "started our hike" in win0.transcript_text and "fresh snow" in win0.transcript_text,
    )
    check(
        "window 0 dedups the near-identical caption (3 unique: trail, forest, panning shot)",
        len(win0.captions) == 3, win0.captions,
    )
    check("window 1 has the lake transcript + caption", "lake" in win1.transcript_text and any("lake" in c for c in win1.captions))
    check("fused_text includes both modalities", "Transcript:" in win0.fused_text and "Visual observations:" in win0.fused_text)
    check(
        "the panning shot (15-25s) lands ONLY in window 0, never split into window 1",
        any("panning shot" in c for c in win0.captions) and not any("panning shot" in c for c in win1.captions),
        (win0.captions, win1.captions),
    )


def test_chunk_fusion_empty_windows_dropped():
    print("\n[text_chunking: empty windows are dropped]")
    segments = [TranscriptSegment(start=0.0, end=5.0, text="hello")]
    # video implicitly has a long tail with no transcript/captions in between -> only windows with content should survive
    chunks = build_fused_chunks(
        segments=segments, frame_captions=[], video_id="vid2",
        window_seconds=20.0, embed_fn=fake_embed_fn, dedup_threshold=0.95,
    )
    check("only the one populated window survives", len(chunks) == 1, chunks)


# ------------------------------------------------------------------
# whisper_utils: hallucination filtering + audio-event summary
# ------------------------------------------------------------------
def test_hallucination_filter():
    print("\n[whisper_utils.filter_hallucinated_segments]")
    thresholds = dict(no_speech_prob_threshold=0.6, avg_logprob_threshold=-1.0, compression_ratio_threshold=2.4)

    confident_speech = {"text": "we hiked to the summit", "no_speech_prob": 0.05, "avg_logprob": -0.3, "compression_ratio": 1.4}
    check("confident speech segment is not hallucinated", not is_hallucinated_segment(confident_speech, **thresholds))

    music_hallucination = {"text": "thank you for watching, don't forget to subscribe", "no_speech_prob": 0.85, "avg_logprob": -1.6, "compression_ratio": 1.8}
    check(
        "high no_speech_prob + low confidence during music -> flagged as hallucinated",
        is_hallucinated_segment(music_hallucination, **thresholds),
    )

    quiet_but_real_speech = {"text": "...and that's the trail marker", "no_speech_prob": 0.7, "avg_logprob": -0.4, "compression_ratio": 1.5}
    check(
        "high no_speech_prob alone (decoder still confident) is NOT enough to drop a segment",
        not is_hallucinated_segment(quiet_but_real_speech, **thresholds),
    )

    repetitive_gibberish = {"text": "yeah yeah yeah yeah yeah yeah yeah yeah", "no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 3.5}
    check(
        "degenerate/repetitive text (high compression ratio) is flagged even with low no_speech_prob",
        is_hallucinated_segment(repetitive_gibberish, **thresholds),
    )

    segments = [confident_speech, music_hallucination, quiet_but_real_speech, repetitive_gibberish]
    kept, n_dropped = filter_hallucinated_segments(segments, **thresholds)
    check("filter_hallucinated_segments drops exactly the 2 hallucinated segments", n_dropped == 2, n_dropped)
    check("filter_hallucinated_segments keeps the 2 real-speech segments in order", kept == [confident_speech, quiet_but_real_speech])


def test_audio_event_summary():
    print("\n[whisper_utils.audio_event_summary]")
    sample_rate = 16000
    duration_sec = 10
    n_samples = sample_rate * duration_sec
    audio = np.zeros(n_samples, dtype=np.float32)

    # seconds 0-4: loud "speech" signal (also declared as a Whisper segment)
    audio[0:4 * sample_rate] = 0.5
    # seconds 4-7: loud but non-speech signal (music/ambient) -- no segment covers it
    audio[4 * sample_rate:7 * sample_rate] = 0.5
    # seconds 7-10: near-zero signal (silence)

    speech_segments = [{"start": 0.0, "end": 4.0, "text": "hello"}]

    events = audio_event_summary(
        audio, sample_rate, speech_segments, frame_seconds=1.0, silence_rms_threshold=0.01,
    )
    check("duration_sec matches the audio length", events["duration_sec"] == duration_sec, events)
    check("speech proportion is ~0.4 (4 of 10 one-second frames)", abs(events["speech"] - 0.4) < 1e-6, events)
    check("music_or_ambient proportion is ~0.3 (3 of 10 frames)", abs(events["music_or_ambient"] - 0.3) < 1e-6, events)
    check("silence proportion is ~0.3 (3 of 10 frames)", abs(events["silence"] - 0.3) < 1e-6, events)
    check("proportions sum to 1.0", abs(events["speech"] + events["music_or_ambient"] + events["silence"] - 1.0) < 1e-6)

    empty_events = audio_event_summary(np.array([], dtype=np.float32), sample_rate, [], frame_seconds=1.0, silence_rms_threshold=0.01)
    check("empty audio doesn't crash and reports zero duration", empty_events["duration_sec"] == 0.0, empty_events)


# ------------------------------------------------------------------
# rag_utils: taxonomy indexing + retrieval
# ------------------------------------------------------------------
def test_rag_retrieval():
    print("\n[rag_utils.TaxonomyRetriever]")
    labels = ["Skiing", "Kayaking", "Bird watching"]
    texts = [
        "Skiing: traveling over snow on skis in winter",
        "Kayaking: paddling a kayak on open water",
        "Bird watching: observing and identifying wild birds",
    ]
    embeddings = fake_embed_fn(texts)
    index = rag_utils.TaxonomyIndex(taxonomy_name="test", labels=labels, texts=texts, embeddings=embeddings)
    retriever = rag_utils.TaxonomyRetriever(index, fake_embed_fn)

    results = retriever.retrieve("We went paddling a kayak on open water this morning", top_k=2)
    top_label = results[0][0]
    check("retrieves Kayaking as top match for a paddling query", top_label == "Kayaking", results)
    check("respects top_k", len(results) == 2)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        index.save(path)
        loaded = rag_utils.TaxonomyIndex.load(path)
        check("round-trips through save/load", loaded.labels == labels)

    # With top_k above the label count (as config.RAG_TOP_K=25 > 22 here),
    # retrieval must offer every label, never a truncated candidate set.
    all_results = retriever.retrieve("anything at all", top_k=25)
    check("top_k >= label count returns all labels as candidates",
          sorted(label for label, _, _ in all_results) == sorted(labels), all_results)


# ------------------------------------------------------------------
# stage 9 video-level ensemble vote (loaded via importlib since
# filename starts with a digit)
# ------------------------------------------------------------------
def load_stage9():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "pipeline" / "09_aggregate_video_labels.py"
    spec = importlib.util.spec_from_file_location("stage9", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage9"] = module
    spec.loader.exec_module(module)
    return module


def write_chunk_labels(tax_dir, model_keys, records_by_model):
    for mk in model_keys:
        (tax_dir / f"{mk}.json").write_text(json.dumps(records_by_model[mk]))


def test_video_aggregation():
    print("\n[stage9: per-model any-chunk union -> cross-model video vote]")
    stage9 = load_stage9()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        activities_dir = tmp / "chunk_labels" / "activities"
        activities_dir.mkdir(parents=True)
        data_dir = tmp / "data"
        (data_dir / "YouTube").mkdir(parents=True)

        model_keys = [e["key"] for e in stage9.config.ENSEMBLE_MODELS]
        assert len(model_keys) == 6, "expected 6 ensemble model entries in config"

        # videoA: "Skiing" in 4/6 models -> survives vote4of6, not vote5of6
        # videoB: "Kayaking" in 2/6 models -> survives vote2of6, not vote3of6
        # videoC: every model processes it but nobody labels anything
        # Every model processes {A, B, C} so all pass the inclusion coverage
        # check (3/4 videos-with-chunks >= 0.5).
        skiing_models = model_keys[:4]
        kayaking_models = model_keys[:2]
        records_by_model = {}
        for mk in model_keys:
            records_by_model[mk] = [
                {"video_id": "videoA", "chunk_index": 0,
                 "labels": ["Skiing"] if mk in skiing_models else []},
                {"video_id": "videoB", "chunk_index": 0,
                 "labels": ["Kayaking"] if mk in kayaking_models else []},
                {"video_id": "videoC", "chunk_index": 0, "labels": []},
            ]
        write_chunk_labels(activities_dir, model_keys, records_by_model)

        # fused chunks: videoA has 10 chunks, B-D one each; videoD has chunks
        # but no model output at all, videoE is in the CSV without chunks
        with open(data_dir / "YouTube" / "fused_chunks_test_vlm.jsonl", "w") as f:
            f.write(json.dumps({"video_id": "videoA", "chunks": [{"chunk_index": i, "fused_text": "x"} for i in range(10)]}) + "\n")
            for vid in ["videoB", "videoC", "videoD"]:
                f.write(json.dumps({"video_id": vid, "chunks": [{"chunk_index": 0, "fused_text": "x"}]}) + "\n")
        videos_csv = tmp / "videos.csv"
        videos_csv.write_text("video_id\n" + "\n".join(["videoA", "videoB", "videoC", "videoD", "videoE"]) + "\n")

        out_dir = tmp / "vote_out"
        stage9.config.CHUNK_LABELS_DIR = tmp / "chunk_labels"
        stage9.config.DATA_DIR = data_dir
        stage9.config.VIDEOS_CSV = videos_csv
        stage9.config.VIDEO_LABELS_DIR = out_dir
        stage9.run("test_vlm")

        vote4 = pd.read_pickle(out_dir / "activities_vote4of6.pkl")
        check("series named activities_vote4of6", vote4.name == "activities_vote4of6", vote4.name)
        check("index follows the CSV order, videoE included",
              list(vote4.index) == ["videoA", "videoB", "videoC", "videoD", "videoE"], list(vote4.index))
        check("videoA keeps Skiing at 4/6", vote4["videoA"] == ["Skiing"], vote4["videoA"])
        check("videoB drops Kayaking at 4/6 (only 2/6 models)", vote4["videoB"] == [], vote4["videoB"])
        check("videoC processed but unlabeled -> []", vote4["videoC"] == [])
        check("videoD (chunks but no model output) -> []", vote4["videoD"] == [])
        check("videoE (CSV only, no chunks) -> []", vote4["videoE"] == [])

        vote2 = pd.read_pickle(out_dir / "activities_vote2of6.pkl")
        check("videoB keeps Kayaking at 2/6", vote2["videoB"] == ["Kayaking"])
        vote6 = pd.read_pickle(out_dir / "activities_vote6of6.pkl")
        check("nothing survives a unanimous 6/6 vote", all(v == [] for v in vote6.tolist()))
        selected_pkl = (out_dir / "activities_selected.pkl").exists()
        check("no _selected.pkl is written (analysis reads voteKofN directly)",
              not selected_pkl, selected_pkl)

        # A model below the inclusion coverage drops out of the vote entirely:
        # cripple model 6 to a single video -> 1/4 coverage < 0.5 -> N becomes 5
        crippled = dict(records_by_model)
        crippled[model_keys[5]] = [{"video_id": "videoC", "chunk_index": 0, "labels": []}]
        write_chunk_labels(activities_dir, model_keys, crippled)
        out_dir2 = tmp / "vote_out_crippled"
        stage9.config.VIDEO_LABELS_DIR = out_dir2
        stage9.run("test_vlm")
        import pathlib
        check("crippled model excluded -> of5 files written",
              (out_dir2 / "activities_vote5of5.pkl").exists()
              and not (out_dir2 / "activities_vote1of6.pkl").exists(),
              sorted(p.name for p in pathlib.Path(out_dir2).glob("*.pkl")))
        vote4of5 = pd.read_pickle(out_dir2 / "activities_vote4of5.pkl")
        check("videoA still keeps Skiing at 4/5 (the 4 skiing models all included)",
              vote4of5["videoA"] == ["Skiing"], vote4of5["videoA"])

        # Single-model design: only poro2_70b (the aitta default) has output
        # -> exactly one model included -> vote1of1 is the consumed product.
        # Remove every other model's labels first so this run is isolated.
        for mk in model_keys[1:]:
            (activities_dir / f"{mk}.json").unlink(missing_ok=True)
        single = {model_keys[0]: records_by_model[model_keys[0]]}
        write_chunk_labels(activities_dir, model_keys[:1], single)
        out_dir3 = tmp / "vote_out_single"
        stage9.config.VIDEO_LABELS_DIR = out_dir3
        stage9.run("test_vlm")
        check("single-model run writes vote1of1",
              (out_dir3 / "activities_vote1of1.pkl").exists(),
              sorted(p.name for p in pathlib.Path(out_dir3).glob("*.pkl")))
        vote1of1 = pd.read_pickle(out_dir3 / "activities_vote1of1.pkl")
        check("single-model union: videoA keeps Skiing, videoB keeps Kayaking",
              vote1of1["videoA"] == ["Skiing"] and vote1of1["videoB"] == ["Kayaking"],
              (vote1of1["videoA"], vote1of1["videoB"]))
        check("no multi-model threshold files in the single-model run",
              not (out_dir3 / "activities_vote1of2.pkl").exists())


def test_expand_long_segments():
    print("\n[shot_detection.expand_long_segments: dense keyframe coverage for cut-free footage]")

    # Regression case: a 10-minute video with zero detected hard cuts (slow
    # drone pans) used to collapse to a single (0, 600) segment -> 1
    # keyframe for the whole video, looking like "no shots detected" despite
    # substantial visual change over time.
    whole_video_fallback = [(0.0, 600.0)]
    expanded = expand_long_segments(whole_video_fallback, target_interval=15.0)
    check("600s video expands into 40 segments of 15s each", len(expanded) == 40, len(expanded))
    check("segments are contiguous and cover the full duration", expanded[0][0] == 0.0 and expanded[-1][1] == 600.0)
    check(
        "no gap between keyframe midpoints exceeds the target interval",
        all(b[1] - b[0] <= 15.0 + 1e-9 for b in expanded),
    )

    # A short segment (a real, tight hard cut) should pass through unchanged.
    short_shots = [(0.0, 3.0), (3.0, 8.0)]
    check("segments shorter than target_interval are left untouched", expand_long_segments(short_shots, 15.0) == short_shots)

    # Mixed: one short real cut plus one long cut-free stretch.
    mixed = [(0.0, 5.0), (5.0, 35.0)]
    expanded_mixed = expand_long_segments(mixed, 15.0)
    check("short segment untouched, long segment split into 2", expanded_mixed == [(0.0, 5.0), (5.0, 20.0), (20.0, 35.0)], expanded_mixed)

    # target_interval <= 0 disables expansion (defensive default).
    check("non-positive target_interval is a no-op", expand_long_segments(whole_video_fallback, 0.0) == whole_video_fallback)

    # subsample_shots still caps the (now denser) keyframe count.
    capped = subsample_shots(expanded, max_keyframes=10)
    check("subsample_shots still enforces the MAX_KEYFRAMES_PER_VIDEO cap after expansion", len(capped) == 10, len(capped))


def test_build_tagged_segments():
    print("\n[shot_detection.build_tagged_segments: shot_index grouping for motion-aware captioning]")

    # Two short real cuts (no subdivision) + one long cut-free shot (gets
    # subdivided) -- mirrors the synthetic-video smoke test used to validate
    # 02_detect_shots_and_extract_keyframes.py end-to-end.
    raw_shots = [(0.0, 5.0), (5.0, 10.0), (10.0, 30.0)]
    tagged = build_tagged_segments(raw_shots, target_interval=15.0)

    check("4 total segments (1 + 1 + 2 from the subdivided long shot)", len(tagged) == 4, tagged)

    by_shot = {}
    for shot_index, start, end in tagged:
        by_shot.setdefault(shot_index, []).append((start, end))

    check("3 distinct shot_index groups", len(by_shot) == 3, by_shot)
    check("short shot 0 has exactly 1 segment", by_shot[0] == [(0.0, 5.0)])
    check("short shot 1 has exactly 1 segment", by_shot[1] == [(5.0, 10.0)])
    check(
        "long cut-free shot 2 (20s > 15s target) is subdivided into 2 segments sharing shot_index=2",
        by_shot[2] == [(10.0, 20.0), (20.0, 30.0)], by_shot[2],
    )

    # Degenerate inputs shouldn't crash.
    check("empty input -> empty output", build_tagged_segments([], 15.0) == [])


def test_group_by_shot():
    print("\n[shot_detection.group_by_shot: bundling frame records for multi-image captioning]")

    # Deliberately out of chronological order and interleaved across shots,
    # to check both the grouping and the within-shot re-sorting by keyframe_ts.
    records = [
        {"shot_index": 0, "keyframe_ts": 2.5, "frame_name": "f0"},
        {"shot_index": 2, "keyframe_ts": 25.0, "frame_name": "f3"},
        {"shot_index": 1, "keyframe_ts": 7.5, "frame_name": "f1"},
        {"shot_index": 2, "keyframe_ts": 15.0, "frame_name": "f2"},  # out of order vs f3 above
    ]
    groups = group_by_shot(records)

    check("3 groups, one per distinct shot_index", len(groups) == 3, groups)
    # shot_index 2 appears (as record f3) before shot_index 1 (record f1) in
    # the input, so first-seen order is 0, 2, 1 -- not numeric order.
    check("first-seen shot order preserved: shot 0, then 2, then 1", [g[0]["shot_index"] for g in groups] == [0, 2, 1], groups)
    check("shot 0 group has its single frame", [r["frame_name"] for r in groups[0]] == ["f0"])
    check(
        "shot 2 group is re-sorted chronologically (f2 @15s before f3 @25s) despite input order",
        [r["frame_name"] for r in groups[1]] == ["f2", "f3"], groups[1],
    )
    check("shot 1 group has its single frame", [r["frame_name"] for r in groups[2]] == ["f1"])

    check("empty input -> empty output", group_by_shot([]) == [])


class _FakeNerPipeline:
    """Stands in for the real HF pipeline: finds a fixed set of substrings
    per call and reports them with the same shape (entity_group/score/
    start/end) redact_texts() expects, so the splicing/windowing logic can
    be checked without downloading a real model."""

    ENTITIES = {"Matti Meikalainen": ("PER", 0.95), "Helsinki": ("LOC", 0.9), "Acme": ("ORG", 0.5)}

    def __call__(self, texts, batch_size=None):
        out = []
        for t in texts:
            found = []
            for word, (group, score) in self.ENTITIES.items():
                idx = t.find(word)
                if idx != -1:
                    found.append({"entity_group": group, "score": score, "start": idx, "end": idx + len(word)})
            out.append(found)
        return out


def test_privacy_ner_windowing():
    print("\n[privacy_ner.split_into_windows]")
    text = "a bb ccc dddd " * 20  # 280 chars, no natural sentence breaks
    windows = split_into_windows(text, max_chars=50)

    check("every window's text matches its offset in the original string",
          all(text[offset:offset + len(piece)] == piece for offset, piece in windows))
    check("no window exceeds max_chars", all(len(piece) <= 50 for _, piece in windows))
    check("windows are never split mid-word (each piece ends on a word boundary or EOF)",
          all(offset + len(piece) == len(text) or text[offset + len(piece)] == " " for offset, piece in windows))
    check("empty string -> no windows", split_into_windows("", 50) == [])


def test_privacy_regex_scrub():
    print("\n[privacy_ner.scrub_structured_pii]")

    texts = [
        "Contact me at matti.m@example.com or see <b>www.example.com/trip</b> \U0001F600",
        "Call 040 123 4567 or visit https://example.fi/photos for the full album!",
        "My PIN is 481920 -- don't share it.",
        "We hiked 15 km today and saw the lake at 3.5 hours in.",
    ]
    scrubbed = scrub_structured_pii(texts, min_phone_digits=6, min_digit_run=4)

    check("email replaced with [EMAIL]", "[EMAIL]" in scrubbed[0] and "matti.m@example.com" not in scrubbed[0])
    check("bare-domain URL replaced with [URL]", "[URL]" in scrubbed[0] and "example.com/trip" not in scrubbed[0])
    check("HTML tags stripped", "<b>" not in scrubbed[0] and "</b>" not in scrubbed[0])
    check("emoji stripped", "\U0001F600" not in scrubbed[0])

    check("https:// URL replaced with [URL]", "[URL]" in scrubbed[1] and "example.fi" not in scrubbed[1])
    check("phone-like number (spaced digits) replaced with [NUMBER]",
          "[NUMBER]" in scrubbed[1] and "123 4567" not in scrubbed[1])

    check("bare 6-digit run (PIN) replaced with [NUMBER]", scrubbed[2] == "My PIN is [NUMBER] -- don't share it.")

    check("short numbers in normal transcript text are left untouched (not over-redacted)",
          scrubbed[3] == texts[3], scrubbed[3])


def test_privacy_ner_redaction():
    print("\n[privacy_ner.redact_texts]")
    texts = [
        "Transcript: Matti Meikalainen walked through Helsinki today.\n"
        "Visual observations: A person near the Acme office.",
        "No entities here at all.",
    ]
    redacted = redact_texts(
        texts, _FakeNerPipeline(), entity_types=["PER", "LOC", "ORG", "MISC"],
        score_threshold=0.8, window_max_chars=800, batch_size=8,
    )

    check("PER entity replaced with [PER]", "[PER]" in redacted[0] and "Matti Meikalainen" not in redacted[0])
    check("LOC entity replaced with [LOC]", "[LOC]" in redacted[0] and "Helsinki" not in redacted[0])
    check("below-threshold ORG entity (score 0.5 < 0.8) left untouched", "Acme" in redacted[0] and "[ORG]" not in redacted[0])
    check("text with no matches is returned unchanged", redacted[1] == texts[1])

    long_text = "Matti Meikalainen was here. " + ("filler word " * 60) + "and later Helsinki too."
    redacted_long = redact_texts(
        [long_text], _FakeNerPipeline(), entity_types=["PER", "LOC", "ORG", "MISC"],
        score_threshold=0.8, window_max_chars=60, batch_size=8,
    )[0]
    check("entities beyond the first window are still redacted when text needs splitting",
          "[PER]" in redacted_long and "[LOC]" in redacted_long, redacted_long)


def load_stage6():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "pipeline" / "06_privacy_scrub_chunks.py"
    spec = importlib.util.spec_from_file_location("stage6", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _HeaderMisclassifyingPipeline:
    """Regression fixture: a real NER model can misclassify the capitalized
    word right after fused_text's "Transcript: "/"Visual observations: "
    headers as an entity, corrupting our own scaffolding (e.g. "Visual
    observations:" -> "[LOC] observations:"). Tags literal "Visual"
    wherever it appears, so the test fails loudly if split_lines() ever
    lets header text reach the model."""

    def __call__(self, texts, batch_size=None):
        out = []
        for t in texts:
            idx = t.find("Visual")
            out.append([{"entity_group": "LOC", "score": 0.99, "start": idx, "end": idx + 6}] if idx != -1 else [])
        return out


def test_privacy_scrub_never_exposes_fused_text_headers():
    print("\n[06_privacy_scrub_chunks: structural headers excluded from NER]")
    stage6 = load_stage6()

    fused_text = "Transcript: We saw something near Visual Lake today.\nVisual observations: A person walks."
    parts = stage6.split_lines(fused_text)

    check("2 lines split out", len(parts) == 2, parts)
    check("line 0 gets TRANSCRIPT_PREFIX, body has no leading prefix text",
          parts[0][0] == "Transcript: " and parts[0][1] == "We saw something near Visual Lake today.", parts[0])
    check("line 1 gets CAPTIONS_PREFIX, body has no leading prefix text",
          parts[1][0] == "Visual observations: " and parts[1][1] == "A person walks.", parts[1])
    check("join_lines(split_lines(x)) round-trips exactly", stage6.join_lines(parts) == fused_text)

    # Run redact_texts() over just the bodies (as main() does): the fake
    # pipeline tags every literal "Visual" it sees, so this checks that the
    # in-body "Visual Lake" gets redacted while the CAPTIONS_PREFIX header
    # -- never passed to the pipeline -- survives untouched.
    bodies = [body for _, body in parts]
    redacted_bodies = redact_texts(
        bodies, _HeaderMisclassifyingPipeline(),
        entity_types=["PER", "LOC", "ORG", "MISC"], score_threshold=0.8,
        window_max_chars=800, batch_size=8,
    )
    rebuilt = stage6.join_lines(list(zip([p[0] for p in parts], redacted_bodies)))

    check("real in-body 'Visual' (part of 'Visual Lake') IS redacted", "[LOC] Lake" in rebuilt, rebuilt)
    check("the structural 'Visual observations:' header survives untouched",
          rebuilt.endswith("Visual observations: A person walks."), rebuilt)


def load_stage8():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "pipeline" / "08_ensemble_label_chunks.py"
    spec = importlib.util.spec_from_file_location("stage8", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage8"] = module
    spec.loader.exec_module(module)
    return module


def test_stage8_prompt_offers_all_22_candidates():
    print("\n[08_ensemble_label_chunks: prompt construction, all 22 candidates]")
    stage8 = load_stage8()
    taxonomy = io_utils.load_taxonomy("activities", config.TAXONOMIES["activities"])
    csv_labels = pd.read_csv(config.TAXONOMIES["activities"]["path"])["activity"].dropna().unique().tolist()
    check("LVVI3 taxonomy has exactly 22 unique activity labels", len(csv_labels) == 22, len(csv_labels))

    # Synthetic retrieval result: every label offered, like RAG_TOP_K=25
    # guarantees (retrieval text == the label itself, no definitions in the CSV).
    candidates = [(label, taxonomy.display_text(label), 0.5) for label in taxonomy.labels]
    messages = stage8.build_messages(
        chunk_text="Transcript: We rented kayaks and paddled across the lake.\n"
                   "Visual observations: A person paddles a yellow kayak.",
        taxonomy_name="activities",
        candidates=candidates,
        video_title="Kayak trip to Päijänne",
        video_description="A weekend paddling trip.",
    )
    system_prompt = messages[0]["content"]
    user_content = messages[1]["content"]

    check("system prompt names the activities taxonomy",
          stage8.TAXONOMY_DESC == {"activities": "outdoor activities"}, stage8.TAXONOMY_DESC)
    check("no experiences taxonomy leaked into the prompt",
          "experience" not in system_prompt.lower(), system_prompt[:200])
    check("every one of the 22 labels appears as a candidate bullet",
          all(f"- {label}:" in system_prompt for label in taxonomy.labels),
          [l for l in taxonomy.labels if f"- {l}:" not in system_prompt])
    check("the creator-performs rule is present",
          "video creator actually performs" in system_prompt)
    check("user content carries title/description/excerpt",
          "Kayak trip to Päijänne" in user_content and "paddled across the lake" in user_content)

    # The full validate-response path: a mocked LLM answer mixing one real
    # LVVI3 label with a hallucinated one parses against the candidates.
    real_label = taxonomy.labels[0]
    fake_response = f'["{real_label}", "Not a real activity"]'
    parsed = io_utils.parse_label_list(fake_response, [c[0] for c in candidates])
    check("response parsing keeps only real LVVI3 labels",
          parsed == [real_label], (parsed, real_label))


def main():
    test_taxonomy_config()
    test_parse_label_list()
    test_chunk_fusion()
    test_chunk_fusion_empty_windows_dropped()
    test_hallucination_filter()
    test_audio_event_summary()
    test_rag_retrieval()
    test_video_aggregation()
    test_expand_long_segments()
    test_build_tagged_segments()
    test_group_by_shot()
    test_privacy_ner_windowing()
    test_privacy_regex_scrub()
    test_privacy_ner_redaction()
    test_privacy_scrub_never_exposes_fused_text_headers()
    test_stage8_prompt_offers_all_22_candidates()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
