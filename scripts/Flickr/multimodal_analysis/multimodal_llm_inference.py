import os
import pandas as pd
import re
import json
from pathlib import Path
from PIL import Image
import argparse

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch



# ---------------- CONFIG ----------------
DATA_DIR = Path("/scratch/project_2006782/leyixu/LVVI3_YouTube/data")
SURVEY_PATH = DATA_DIR / "LVVI3" / "LVVI3_activities_inference.csv"
INPUT_DIR= DATA_DIR / "Flickr" / "multimodal_json_input"
OUTPUT_DIR= DATA_DIR / "Flickr" / "inference_json_output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path("/scratch/project_2006782/leyixu/models/Qwen3-VL-8B-Instruct")
DEVICE = "cuda:0"
DTYPE = torch.float16


# ---------------- Load Qwen3-VL ----------------
def load_model(model_path):

    print("Loading model...")

    processor = AutoProcessor.from_pretrained(model_path)

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )

    model.eval()

    print("Model loaded.")

    return model, processor


# ---------------- Build prompt ----------------
def build_prompt(image, survey_list):
    survey_str = ", ".join(survey_list)

    prompt = f"""
You are an image analysis system for nature-based tourism and outdoor recreation.

Your task is to analyze multimodal image data (text + visual) and extract structured information.

========================
TASK OVERVIEW
========================

Step 1 — Human detection:
- Determine whether humans are present in the image.
- Output:
  - 1 = human present
  - 0 = no human present

Step 2 — Information extraction from multimodal content:
Extract:
1. Outdoor recreation activities
2. Accompany type (social composition)
3. Demographics (age group and gender)


========================
INPUT DATA
========================

TEXTUAL DATA:
- Title: {image.get('textual', {}).get("title", "")}
- Description: {image.get('textual', {}).get("description", "")}
- Tags: {image.get('textual', {}).get("tags", "")}

VISUAL DATA:
- Image will be provided in the image content

MODALITY WEIGHTS:
{image.get('modality_weights', "")}

REFERENCE LABELS
------------------------

Accompany types:
- Alone
- With family
- With friends
- With work colleagues/business guests
- With school class/student group
- With club or association
- With other group
- Unknown

Age groups:
- <16
- 16-24
- 25-34
- 35-44
- 45-54
- 55-64
- 65-74
- >75
- Unknown

Gender:
- Male
- Female
- Unknown

Survey activity list:
[{survey_str}]

========================
RULES
========================

1. Activity labeling:
   - If activity matches the survey list → type = "survey_activity"
   - Otherwise → type = "new_activity"
   - Only include clearly visible physical outdoor recreation activities

2. Multiple activities:
   - Detect ALL activities visible in the image or mentioned in the text

3. Modality use:
   - Use both textual and visual inputs

4. Missing information:
   - If uncertain, use "Unknown"
   - If no activity visible → return empty activity list

5. Output must be valid JSON:
   - Double quotes only
   - No trailing commas
   - No unescaped characters

========================
OUTPUT FORMAT
========================

Return ONLY valid JSON:

{{
  "image_id": {image.get("image_id", "")},
  "human_presence": 1 | 0,

  "activities": [
    {{
      "name": "activity name",
      "type": "survey_activity | new_activity",
      "confidence": 0.00,
      "modality_source": ["textual", "visual"]
    }}
  ],

  "accompany_type": "Alone | With family | With friends | With work colleagues/business guests | With school class/student group | With club or association | With other group | Unknown",

  "age_group": "<16 | 16-24 | 25-34 | 35-44 | 45-54 | 55-64 | 65-74 | >75 | Unknown",

  "gender": "Male | Female | Unknown",
  
  "inference_metadata": {{
    "model": "Qwen3-VL",
    "temperature": 0
    }}
}}
"""
    return prompt

# ---------------- JSON PARSER ----------------
def extract_json(text):
    if isinstance(text, list):
        text = text[0]

    text = re.sub(r"```json|```", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found")

    brace = 0
    in_string = False
    escape = False
    end = -1

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue

        if not in_string:
            if c == "{":
                brace += 1
            elif c == "}":
                brace -= 1
                if brace == 0:
                    end = i + 1
                    break

    if end == -1:
        raise ValueError("Unmatched JSON braces")

    json_str = text[start:end]

    return json.loads(json_str)


# ---------------- INFERENCE ----------------
def infer_image(model, processor, image, survey_list):

    img_path = image.get("visual", {}).get("photo_url", "")

    prompt = build_prompt(image, survey_list)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0
        )

    gen_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated)
    ]

    output_text = processor.batch_decode(
        gen_trimmed,
        skip_special_tokens=True
    )

    print("RAW:", output_text)

    return extract_json(output_text)


# ---------------- PROCESS IMAGE ----------------
def process_image(model, processor, json_path, output_dir):

    df_survey = pd.read_csv(SURVEY_PATH)
    survey_list = df_survey.sub_activity.tolist()

    with open(json_path, "r", encoding="utf-8") as f:
        image_input = json.load(f)

    image_id = image_input.get("image_id", "")

    try:
        output = infer_image(model, processor, image_input, survey_list)

        out_path = output_dir / f"{image_id}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Failed {image_id}: {e}")


# ---------------- MAIN ----------------
def main():

    image_list = [Path(f).stem for f in os.listdir(INPUT_DIR)][16270:]
    json_files = [INPUT_DIR / f"{img}.json" for img in image_list]

    model, processor = load_model(MODEL_PATH)

    for json_file in json_files:
        process_image(model, processor, json_file, OUTPUT_DIR)


if __name__ == "__main__":
    main()