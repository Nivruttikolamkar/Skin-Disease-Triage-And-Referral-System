import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
BASE = r"C:\Users\VICTUS\Videos\Final_year_project"
METADATA_PATH = os.path.join(BASE, "METADATA")
DATASET_PATH = os.path.join(BASE, "DATASET")
MIN_IMAGES_PER_CLASS = 80
VAL_SPLIT_RATIO = 0.20  # 20% of train becomes validation
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")

OUTPUT_DIR = BASE   # <-- everything saved directly in root now, no subfolder
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_disk_lookup():
    print("Scanning DATASET folder for images...")
    lookup = {}
    for root, dirs, files in os.walk(DATASET_PATH):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                lookup[f] = os.path.join(root, f)
    print(f"Found {len(lookup)} images on disk.\n")
    return lookup


def load_and_filter():
    train_df = pd.read_csv(os.path.join(METADATA_PATH, "train_split-1.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_PATH, "test_split-1.csv"))

    keep_cols = ["Image_name", "Disease_label", "Main_class", "Sub_class", "Age", "Sex", "Fitzpatrick"]
    train_df = train_df[keep_cols].copy()
    test_df = test_df[keep_cols].copy()

    combined = pd.concat([train_df, test_df], ignore_index=True)
    class_counts = combined["Disease_label"].value_counts()
    selected_classes = class_counts[class_counts >= MIN_IMAGES_PER_CLASS].index.tolist()

    print(f"Selected {len(selected_classes)} classes with >= {MIN_IMAGES_PER_CLASS} images:\n")
    for cls in selected_classes:
        print(f"  {cls}: {class_counts[cls]}")

    train_df = train_df[train_df["Disease_label"].isin(selected_classes)].reset_index(drop=True)
    test_df = test_df[test_df["Disease_label"].isin(selected_classes)].reset_index(drop=True)

    print(f"\nFiltered train rows: {len(train_df)}")
    print(f"Filtered test rows: {len(test_df)}")

    return train_df, test_df, selected_classes


def add_filepaths(df, disk_lookup):
    df["filepath"] = df["Image_name"].map(disk_lookup)
    missing = df["filepath"].isna().sum()
    if missing > 0:
        print(f"[!] Warning: {missing} rows have no matching image on disk — dropping them")
        df = df.dropna(subset=["filepath"]).reset_index(drop=True)
    return df


def make_val_split(train_df):
    train_final, val_final = train_test_split(
        train_df,
        test_size=VAL_SPLIT_RATIO,
        stratify=train_df["Disease_label"],
        random_state=42
    )
    return train_final.reset_index(drop=True), val_final.reset_index(drop=True)


def plot_class_distribution(df, title, filename):
    counts = df["Disease_label"].value_counts()
    plt.figure(figsize=(12, 8))
    counts.plot(kind="barh")
    plt.xlabel("Number of images")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved chart: {save_path}")


if __name__ == "__main__":
    disk_lookup = build_disk_lookup()
    train_df, test_df, selected_classes = load_and_filter()

    train_df = add_filepaths(train_df, disk_lookup)
    test_df = add_filepaths(test_df, disk_lookup)

    train_final, val_final = make_val_split(train_df)

    print(f"\n===== FINAL SPLIT SIZES =====")
    print(f"Train: {len(train_final)}")
    print(f"Validation: {len(val_final)}")
    print(f"Test: {len(test_df)}")

    train_final.to_csv(os.path.join(OUTPUT_DIR, "train_final.csv"), index=False)
    val_final.to_csv(os.path.join(OUTPUT_DIR, "val_final.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_final.csv"), index=False)

    print(f"\nSaved: train_final.csv, val_final.csv, test_final.csv in {OUTPUT_DIR}")

    plot_class_distribution(train_final, "Train Set - Class Distribution", "train_class_distribution.png")
    plot_class_distribution(val_final, "Validation Set - Class Distribution", "val_class_distribution.png")
    plot_class_distribution(test_df, "Test Set - Class Distribution", "test_class_distribution.png")

    with open(os.path.join(OUTPUT_DIR, "class_list.txt"), "w") as f:
        for cls in sorted(selected_classes):
            f.write(cls + "\n")
    print(f"Saved class list ({len(selected_classes)} classes) to class_list.txt")