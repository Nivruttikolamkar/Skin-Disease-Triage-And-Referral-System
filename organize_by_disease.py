"""Organize dataset images into per-disease folders for easy browsing/reporting."""
import os
import shutil
import pandas as pd

BASE_DIR = r"C:\Users\VICTUS\Videos\Final_year_project"
TRAIN_CSV = os.path.join(BASE_DIR, "train_consolidated.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_consolidated.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_by_disease")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Combine train + val so every image is included
all_df = pd.concat([train_df, val_df], ignore_index=True)

print(f"Total images to organize: {len(all_df)}")

copied = 0
skipped = 0

for _, row in all_df.iterrows():
    src_path = row["filepath"]
    disease = row["Disease_label"]

    # Make a safe folder name (remove special characters like &, /)
    safe_disease_name = disease.replace("&", "and").replace("/", "-").strip()
    disease_folder = os.path.join(OUTPUT_DIR, safe_disease_name)
    os.makedirs(disease_folder, exist_ok=True)

    if not os.path.exists(src_path):
        skipped += 1
        continue

    filename = os.path.basename(src_path)
    dest_path = os.path.join(disease_folder, filename)

    try:
        shutil.copy2(src_path, dest_path)
        copied += 1
    except Exception as e:
        print(f"Failed to copy {src_path}: {e}")
        skipped += 1

print(f"\nDone! Copied: {copied}, Skipped (not found): {skipped}")
print(f"Organized images are in: {OUTPUT_DIR}")
