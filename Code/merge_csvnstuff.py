import os
import pandas as pd
import json
from glob import glob

# Adjust this to your actual chunks folder
CHUNK_FOLDER = r"C:\Users\parth\OneDrive\Desktop\Parth\SJSU\Coursework\297\Dataset\Fakeddit\chunks"
OUTPUT_FOLDER = r"C:\Users\parth\OneDrive\Desktop\Parth\SJSU\Coursework\297\Dataset\Fakeddit\chunks\merged"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

datasets = ["all_train", "all_validate", "all_test_public"]

all_frames = []
all_missing_ids = set()

for dataset in datasets:
    print(f"\n🔄 Merging dataset: {dataset}")

    # === Merge CSVs ===
    csv_files = sorted(glob(os.path.join(CHUNK_FOLDER, f"extracted_chunk_{dataset}.tsv_part_*.csv")))
    if not csv_files:
        print(f"⚠️ No CSV files found for {dataset}")
        continue

    frames = []
    columns = None

    for f in csv_files:
        if os.path.getsize(f) > 0:
            try:
                df = pd.read_csv(f)
                frames.append(df)
                if columns is None:
                    columns = df.columns
            except pd.errors.EmptyDataError:
                print(f"⚠️ Skipping corrupt or truly empty file: {f}")
                if columns is not None:
                    frames.append(pd.DataFrame(columns=columns))
        else:
            print(f"⚠️ Empty file with no data: {f}")
            if columns is not None:
                frames.append(pd.DataFrame(columns=columns))

    if not frames:
        print(f"⚠️ No usable data found for {dataset}")
        continue

    df_dataset = pd.concat(frames, ignore_index=True)
    print(f"✅ Merged {len(csv_files)} CSV parts: {len(df_dataset)} total rows")

    # === Merge missing posts ===
    json_files = sorted(glob(os.path.join(CHUNK_FOLDER, f"missing_posts_chunk_{dataset}.tsv_part_*.json")))
    missing_ids = set()
    for jf in json_files:
        try:
            with open(jf) as f:
                ids = json.load(f)
                missing_ids.update(ids)
        except Exception as e:
            print(f"⚠️ Could not read {jf}: {e}")

    print(f"📄 Found {len(json_files)} missing post files ({len(missing_ids)} unique IDs)")
    all_missing_ids.update(missing_ids)

    # === Append fallback marker
    df_dataset["used_fallback"] = df_dataset["post_id"].isin(missing_ids)
    all_frames.append(df_dataset)

# === Final merge of all datasets
final_df = pd.concat(all_frames, ignore_index=True)
final_df["used_fallback"] = final_df["post_id"].isin(all_missing_ids)

# === Save single output CSV and JSON
merged_csv = os.path.join(OUTPUT_FOLDER, "merged_dataset.csv")
final_df.to_csv(merged_csv, index=False)
print(f"\n💾 Saved final merged CSV: {merged_csv}")

merged_json = os.path.join(OUTPUT_FOLDER, "merged_missing_posts.json")
with open(merged_json, "w") as f:
    json.dump(sorted(all_missing_ids), f)
print(f"💾 Saved combined missing posts list: {merged_json}")

print("\n✅ All datasets merged into a single file and fallback flagged.")
