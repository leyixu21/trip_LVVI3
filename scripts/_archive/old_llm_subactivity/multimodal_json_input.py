import pandas as pd
import json
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_DIR = Path("/scratch/project_2006782/leyixu/LVVI3_YouTube/data")
CSV_PATH = DATA_DIR / "Flickr" / "flickr_finland_state_lands.csv"
OUTPUT_ROOT = DATA_DIR / "Flickr" / "multimodal_json_input"

OUTPUT_ROOT.mkdir(exist_ok=True)

# ---------------- Prepare input json ----------------
def prepare_input_json(image):
    image_id = image["id"]
    print("Processing", image_id)

    image_json = {
        "image_id": image_id,
        "textual": {
            "title": image["title"],
            "description": image["description"],
            "tags": image["tags"]
        },
        "visual": {
            "image_url": image['photo_url']
        },
        "modality_weights": {
            "text": 0.5,
            "visual": 0.5
        }
    }

    with open(OUTPUT_ROOT / f"{image_id}.json", "w", encoding="utf-8") as f:
        json.dump(image_json, f, indent=2, ensure_ascii=False)


# ---------------- Main ----------------
def main():
    df_images = pd.read_csv(CSV_PATH)
    
    print(f"Found {len(df_images)} matching images in CSV")
    
    for _, image in df_images.iterrows():
        prepare_input_json(image)

if __name__ == "__main__":
    main()

