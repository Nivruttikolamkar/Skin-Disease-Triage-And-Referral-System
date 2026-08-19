import os
import pandas as pd
import tensorflow as tf

BASE = r"C:\Users\VICTUS\Videos\Final_year_project"
files_to_check = ["train_final.csv", "val_final.csv", "test_final.csv"]

def is_valid_image(filepath):
    try:
        img_bytes = tf.io.read_file(filepath)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        # force actual computation (lazy ops can hide errors until used)
        _ = img.shape
        _ = tf.image.resize(img, (224, 224)).numpy()
        return True
    except Exception as e:
        print(f"[BAD IMAGE] {filepath} -> {e}")
        return False


for filename in files_to_check:
    path = os.path.join(BASE, filename)
    df = pd.read_csv(path)

    print(f"\nChecking {filename} ({len(df)} images)...")
    valid_mask = df["filepath"].apply(is_valid_image)

    num_bad = (~valid_mask).sum()
    print(f"{filename}: {num_bad} bad images out of {len(df)}")

    if num_bad > 0:
        df_clean = df[valid_mask].reset_index(drop=True)
        df_clean.to_csv(path, index=False)
        print(f"Cleaned and overwritten: {path} (now {len(df_clean)} rows)")
    else:
        print("No bad images found, file unchanged.")

print("\n✅ TensorFlow-level validation complete for all 3 CSVs.")