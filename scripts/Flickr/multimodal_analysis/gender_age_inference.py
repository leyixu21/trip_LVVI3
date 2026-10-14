import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

from deepface import DeepFace

# ───────────────────────── CPU CONFIG ─────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DEEPFACE_HOME"] = "/scratch/project_2006782/leyixu/model/.deepface"

# ───────────────────────── CONFIG ─────────────────────────────
DATA_DIR = Path("/scratch/project_2006782/leyixu/LVVI3_YouTube/data")

FLICKR_DATA = DATA_DIR / "Flickr" / "flickr_finland_state_lands.csv"

INFERENCE_DIR = DATA_DIR / "Flickr" / "inference_json_output"
OUTPUT_DIR = DATA_DIR / "Flickr" / "gender_age_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# detector
DETECTOR_BACKEND = "yunet"   # alternatives: "retinaface", "mtcnn"

ACTIONS = ["age", "gender"]

IMAGE_TRY_ORDER = [1, 0, 2]

# ───────────────────────── FILTER PARAMETERS ─────────────────────────
FACE_CONF_THRESHOLD = 0.90     # key fix for false positives
MIN_FACE_AREA = 2500           # filter tiny “fake faces”

# ───────────────────────── LOAD DATA ───────────────────────────
df_flickr = pd.read_csv(FLICKR_DATA)
image_ids = df_flickr.id.tolist()



# ───────────────────────── TF CPU ONLY ─────────────────────────
def configure_cpu():
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

configure_cpu()


# ───────────────────────── HELPERS ─────────────────────────────
def load_image_url(image_id: int):
    row = df_flickr[df_flickr["id"] == image_id]
    if row.empty:
        return None
    return row.iloc[0]["photo_url"]


def analyse_image(image_path: str):
    """
    Run DeepFace safely with stricter detection.
    """
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=ACTIONS,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,   # 🔥 important change
            silent=True
        )

        if isinstance(result, dict):
            result = [result]

        return result

    except Exception:
        return []


def is_valid_face(r: dict) -> bool:
    """
    Strong filtering to remove false positives.
    """
    conf = r.get("face_confidence", 0)

    if conf is None:
        return False

    if conf < FACE_CONF_THRESHOLD:
        return False

    region = r.get("region", {}) or {}
    w = region.get("w", 0)
    h = region.get("h", 0)

    if w * h < MIN_FACE_AREA:
        return False

    return True


def extract_face_info(r: dict, filename: str, idx: int):
    gender_raw = r.get("dominant_gender", "Unknown")
    gender_scores = r.get("gender", {})

    conf = gender_scores.get(gender_raw, 0.0)
    if isinstance(conf, (int, float)):
        conf = round(float(conf) / 100.0, 4)

    return {
        "image": filename,
        "face_index": idx,
        "age": int(r.get("age", -1)),
        "gender": gender_raw,
        "gender_confidence": conf,
        "face_confidence": r.get("face_confidence", None),
    }


def process_image(image_id: int):
    """
    Process one image with face filtering.
    """

    out_file = OUTPUT_DIR / f"{image_id}.json"
    if out_file.exists():
        return f"{image_id} skipped"

    image_url = load_image_url(image_id)
    if not image_url:
        return f"{image_id} missing URL"

    # Optional upstream detection gate (kept as-is)
    if not (INFERENCE_DIR / f"{image_id}.json").exists():
        return f"{image_id} no person detection"

    results = analyse_image(image_url)

    faces = []
    for i, r in enumerate(results):
        if is_valid_face(r):   # 🔥 filtering added here
            faces.append(extract_face_info(r, image_id, i))

    output = {
        "image_id": image_id,
        "has_face": len(faces) > 0,
        "num_faces": len(faces),
        "faces": faces,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(output), f, indent=2, ensure_ascii=False)

    return f"{image_id} done ({len(faces)} faces)"


def make_json_serializable(value):
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(make_json_serializable(v) for v in value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    return value


# ───────────────────────── MAIN ────────────────────────────────
def main():
    print(f"Processing {len(image_ids)} images...")

    for image_id in image_ids:
        result = process_image(int(image_id))
        print(result)


if __name__ == "__main__":
    main()