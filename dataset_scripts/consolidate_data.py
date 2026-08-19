import os
import pandas as pd

BASE = r"C:\Users\VICTUS\Videos\Final_year_project"

# Map old 13 granular classes to 8 consolidated broad categories
CONSOLIDATION_MAP = {
    "Acne": "Acne",
    "Alopecia Areata": "Alopecia Areata",
    "Candidal Intertrigo": "Candidal Intertrigo",
    "Contact Dermatitis": "Eczema & Dermatitis",
    "Eczema": "Eczema & Dermatitis",
    "Melasma": "Melasma",
    "Psoriasis": "Psoriasis",
    "Scabies": "Scabies",
    "Steroid Modified Tinea": "Tinea (Fungal Infection)",
    "Tinea Corporis": "Tinea (Fungal Infection)",
    "Tinea Cruris": "Tinea (Fungal Infection)",
    "Tinea Faciei": "Tinea (Fungal Infection)",
    "Vitiligo": "Vitiligo"
}

print("Consolidating datasets...")

for split in ["train", "val", "test"]:
    csv_path = os.path.join(BASE, f"{split}_final.csv")
    df = pd.read_csv(csv_path)
    
    # Map old labels to new consolidated labels
    df["Disease_label"] = df["Disease_label"].str.strip().map(CONSOLIDATION_MAP)
    
    # Save to new CSV
    out_path = os.path.join(BASE, f"{split}_consolidated.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path} ({len(df)} rows)")

# Save new class list sorted alphabetically
classes = sorted(list(set(CONSOLIDATION_MAP.values())))
class_list_path = os.path.join(BASE, "class_list_consolidated.txt")
with open(class_list_path, "w") as f:
    for c in classes:
        f.write(f"{c}\n")

print(f"\n✅ Saved new consolidated classes to {class_list_path}:")
for i, c in enumerate(classes):
    print(f"  {i}: {c}")