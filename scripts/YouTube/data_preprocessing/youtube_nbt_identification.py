import pandas as pd
import re
import time
import argparse

from openai import OpenAI


# -------------------
# Config
# -------------------
MODEL_NAME = "Qwen3-14B-Q4_0.gguf"

# -------------------
# Prompt
# -------------------
R1_SYSTEM_PROMPT = f"""
You are an expert in analyzing YouTube video content to identify nature-based tourism (NBT) and outdoor recreation (OR) related material.
Your task is to evaluate the provided video title and description, and determine if it pertains to NBT or OR. If it does, respond with "Yes"; if not, respond with "No".
"""

# -------------------
# Load the qwen model
# -------------------
openai_api_key = "MobiconRules"
openai_api_base = "http://fip-128-214-253-250.kaj.poutavm.fi:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# -------------------
# Identify NBT content using the qwen model
# -------------------
def identify_nbt_content(combined_text):
    response = client.chat.completions.create(
        model= MODEL_NAME,
        messages = [
        {"role": "system", "content": R1_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the input text: {combined_text} /no_think"}
        ]
    )
    result = response.choices[0].message.content
    # Strip Qwen <think> tags
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result

# -------------------
# Process in batches
# -------------------
def main(start_idx):
    end_idx = start_idx + 2000
    print(f"Reading data rows {start_idx}–{end_idx-1}...")
    df = pd.read_csv('videos_info_finland_ver01.csv')

    df = df.iloc[start_idx:end_idx]

    df["combined_text"] = (
        "Video title: " + df["title"].fillna("") + " " +
        "Video description: " + df["description"].fillna("")
    )

    nbt_series = pd.DataFrame(columns=['video_id', 'nbt_relevance'])

    print(f"Identifying NBT content for {len(df)} videos...")
    start_time = time.time()
    n = len(df)
    for progress, (i, item) in df.iterrows():
        result = identify_nbt_content(item['combined_text'])
        video_id = item['video_id']
        nbt_series.loc[i] = [video_id, result]

        if progress % 50 == 0 or progress == n:
            percent = (progress / n) * 100
            print(f"\rProcessed {progress}/{n} ({percent:.1f}%)", end="")

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nProcessing took {int(hours)}h {int(minutes)}m {seconds:.1f}s")

    output_file = f"nbt_identification_files/nbt_identification_qwen3_14b_{end_idx}.pkl"
    nbt_series.to_pickle(output_file)
    print(f"✅ Results saved to {output_file}")

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a batch of videos for NBT content")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    args = parser.parse_args()

    main(args.start)