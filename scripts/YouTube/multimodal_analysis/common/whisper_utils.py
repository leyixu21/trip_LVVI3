"""Whisper post-processing: hallucination filtering + a lightweight
speech/music/silence audio-event summary.

Pure numpy, no model/GPU dependency, so it's testable off-cluster.
"""
import subprocess

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM via ffmpeg.

    Deliberately reimplemented here instead of importing whisper.audio: that
    submodule is an internal implementation detail of the `openai-whisper`
    package, not a stable public API, and a same-named but unrelated PyPI
    package (`pip install whisper`, as opposed to `openai-whisper`) can
    shadow it -- `import whisper` still succeeds in that case (so
    `whisper.load_model` fails with its own, separate error) but
    `import whisper.audio` fails immediately with "'whisper' is not a
    package". Decoding audio ourselves removes that failure mode entirely.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sample_rate),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed to decode audio from {path}: {stderr}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def is_hallucinated_segment(
    segment: dict,
    no_speech_prob_threshold: float,
    avg_logprob_threshold: float,
    compression_ratio_threshold: float,
) -> bool:
    """A segment is treated as hallucinated if Whisper itself flagged it as
    likely non-speech AND the decoder had low confidence in its own output
    (music/ambient stretches Whisper filled with invented text), or if the
    text is degenerate/repetitive (high compression ratio -> gibberish)."""
    no_speech_prob = segment.get("no_speech_prob", 0.0)
    avg_logprob = segment.get("avg_logprob", 0.0)
    compression_ratio = segment.get("compression_ratio", 0.0)

    low_confidence_no_speech = no_speech_prob > no_speech_prob_threshold and avg_logprob < avg_logprob_threshold
    repetitive_gibberish = compression_ratio > compression_ratio_threshold
    return low_confidence_no_speech or repetitive_gibberish


def filter_hallucinated_segments(
    segments: list[dict],
    no_speech_prob_threshold: float,
    avg_logprob_threshold: float,
    compression_ratio_threshold: float,
) -> tuple[list[dict], int]:
    """Returns (kept_segments, n_dropped)."""
    kept = [
        s for s in segments
        if not is_hallucinated_segment(s, no_speech_prob_threshold, avg_logprob_threshold, compression_ratio_threshold)
    ]
    return kept, len(segments) - len(kept)


def audio_event_summary(
    audio: np.ndarray,
    sample_rate: int,
    speech_segments: list[dict],
    frame_seconds: float,
    silence_rms_threshold: float,
) -> dict:
    """Proportions of the video's duration spent in speech (per Whisper's
    surviving segments), music/ambient sound, and silence -- classified by
    frame-level RMS energy for the non-speech portion. Reuses the same
    decoded audio Whisper already read, so it costs no extra decoding."""
    total_duration = len(audio) / sample_rate if sample_rate else 0.0
    if total_duration <= 0:
        return {"speech": 0.0, "music_or_ambient": 0.0, "silence": 0.0, "duration_sec": 0.0}

    frame_len = max(int(frame_seconds * sample_rate), 1)
    n_frames = int(np.ceil(len(audio) / frame_len))
    if n_frames == 0:
        return {"speech": 0.0, "music_or_ambient": 0.0, "silence": 0.0, "duration_sec": total_duration}

    speech_mask = np.zeros(n_frames, dtype=bool)
    for seg in speech_segments:
        start_frame = max(int(seg["start"] / frame_seconds), 0)
        end_frame = min(int(np.ceil(seg["end"] / frame_seconds)), n_frames)
        speech_mask[start_frame:end_frame] = True

    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        if frame.size:
            rms[i] = np.sqrt(np.mean(np.square(frame)))

    silence_mask = (~speech_mask) & (rms < silence_rms_threshold)
    music_mask = (~speech_mask) & (rms >= silence_rms_threshold)

    return {
        "speech": float(speech_mask.mean()),
        "music_or_ambient": float(music_mask.mean()),
        "silence": float(silence_mask.mean()),
        "duration_sec": total_duration,
    }
