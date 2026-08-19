import os
import pandas as pd

# Hardcoded project root — update this if your project folder ever moves
BASE = r"C:\Users\VICTUS\Videos\Final_year_project"
DATASET_PATH = os.path.join(BASE, "DATASET")
METADATA_PATH = os.path.join(BASE, "METADATA")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")


def explore_metadata():
    print("\n===== METADATA FOLDER =====")
    files = os.listdir(METADATA_PATH)
    print("Files found:", files)

    main_df = None
    for f in files:
        path = os.path.join(METADATA_PATH, f)
        if f.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif f.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            continue

        print(f"\n--- {f} ---")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        print(df.head())

        # treat the file with Disease_label column as the main metadata file
        if "Disease_label" in df.columns:
            main_df = df

    return main_df


def explore_dataset_folders():
    print("\n===== DATASET FOLDER (recursive) =====")
    all_images = []   # list of (filename, full_path)

    for root, dirs, files in os.walk(DATASET_PATH):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                all_images.append((f, os.path.join(root, f)))

    print(f"Total image files found (recursively): {len(all_images)}")
    if all_images:
        print("Sample filenames:", [f for f, _ in all_images[:5]])

    return all_images


def cross_check(df, all_images):
    print("\n===== CROSS-CHECK: metadata vs actual images =====")
    print("Rows in metadata CSV:", len(df))
    print("Image files found on disk:", len(all_images))

    print("\nSample Image_name values from CSV:")
    print(df["Image_name"].head(5).tolist())

    # Build lookup: filename (with and without extension) -> full path
    disk_lookup = {}
    for fname, fpath in all_images:
        name_no_ext = os.path.splitext(fname)[0]
        disk_lookup[name_no_ext] = fpath
        disk_lookup[fname] = fpath

    matched = 0
    unmatched_samples = []
    for name in df["Image_name"]:
        name_str = str(name)
        name_no_ext = os.path.splitext(name_str)[0]
        if name_str in disk_lookup or name_no_ext in disk_lookup:
            matched += 1
        else:
            if len(unmatched_samples) < 5:
                unmatched_samples.append(name_str)

    print(f"\nMatched: {matched} / {len(df)} metadata rows found an image on disk")
    if unmatched_samples:
        print("Sample UNMATCHED Image_name values:", unmatched_samples)


def disease_label_counts(df):
    print("\n===== DISEASE_LABEL VALUE COUNTS =====")
    counts = df["Disease_label"].value_counts()
    print("Total unique diseases:", df["Disease_label"].nunique())
    print(counts.to_string())

    out_path = os.path.join(BASE, "disease_label_counts.csv")
    counts.to_csv(out_path, header=["image_count"])
    print(f"\nSaved full breakdown to: {out_path}")


if __name__ == "__main__":
    df = explore_metadata()
    all_images = explore_dataset_folders()

    if df is not None and all_images:
        cross_check(df, all_images)
        disease_label_counts(df)
    elif df is None:
        print("\n[!] Could not find a metadata file containing 'Disease_label' column.")
    elif not all_images:
        print("\n[!] No images found in DATASET folder — check IMG_EXT or folder path.")