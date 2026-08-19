import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import IMG_SIZE

# ===== CONFIGURATION =====
BASE_DIR = r"C:\Users\VICTUS\Videos\Final_year_project"

# Har naye run ke liye ye tag badal dena (e.g. "v1_confirm", "round3")
RUN_TAG = "v1_confirm"

MODEL_PATH = os.path.join(BASE_DIR, "final_model_consolidated.keras")
EVAL_CSV = os.path.join(BASE_DIR, "val_consolidated.csv")
CLASS_LIST_FILE = os.path.join(BASE_DIR, "class_list_consolidated.txt")

BATCH_SIZE = 16

# ===== LOAD CLASS NAMES =====
with open(CLASS_LIST_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

class_to_idx = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)
print(f"Loaded {num_classes} classes: {class_names}")

# ===== LOAD DATA =====
eval_df = pd.read_csv(EVAL_CSV)
eval_df["label_idx"] = eval_df["Disease_label"].map(class_to_idx)

if eval_df["label_idx"].isnull().any():
    bad = eval_df[eval_df["label_idx"].isnull()]["Disease_label"].unique()
    raise ValueError(f"Found labels in {EVAL_CSV} not present in class list: {bad}")

print(f"Evaluating on {len(eval_df)} samples.")

# ===== PREPROCESSING (must match training) =====
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

def create_eval_dataset(df):
    filepaths = df["filepath"].values
    labels = df["label_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

eval_ds = create_eval_dataset(eval_df)

# ===== LOAD MODEL =====
print(f"\nLoading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)

# ===== RUN PREDICTIONS =====
print("\nRunning predictions...")
y_true = eval_df["label_idx"].values
y_probs = model.predict(eval_ds, verbose=1)
y_pred = np.argmax(y_probs, axis=1)
y_confidence = np.max(y_probs, axis=1)

# ===== OVERALL METRICS =====
overall_acc = accuracy_score(y_true, y_pred)
print("\n" + "="*60)
print(f"OVERALL TEST ACCURACY ({RUN_TAG}): {overall_acc*100:.2f}%")
print("="*60)

print("\nClassification Report:\n")
report = classification_report(
    y_true, y_pred, target_names=class_names, digits=3
)
print(report)

# ===== SAVE VERSIONED REPORT =====
report_path = os.path.join(BASE_DIR, f"evaluation_report_{RUN_TAG}.txt")
with open(report_path, "w") as f:
    f.write(f"Run tag: {RUN_TAG}\n")
    f.write(f"Overall Accuracy: {overall_acc*100:.2f}%\n\n")
    f.write(report)
print(f"\nSaved text report to: {report_path}")

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix [{RUN_TAG}] (Acc: {overall_acc*100:.2f}%)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(BASE_DIR, f"confusion_matrix_{RUN_TAG}.png")
plt.savefig(cm_path, dpi=150)
print(f"Saved confusion matrix plot to: {cm_path}")
plt.close()

# ===== CONFIDENCE-BASED REFERRAL ANALYSIS =====
print("\n" + "="*60)
print("CONFIDENCE-BASED REFERRAL BREAKDOWN")
print("="*60)

referral_lines = []
thresholds = [0.5, 0.6, 0.7, 0.8]
for t in thresholds:
    mask = y_confidence >= t
    n_covered = mask.sum()
    if n_covered == 0:
        line = f"Threshold {t:.1f}: no predictions meet this confidence."
        print(line)
        referral_lines.append(line)
        continue
    acc_at_t = accuracy_score(y_true[mask], y_pred[mask])
    coverage = n_covered / len(y_true) * 100
    line = (
        f"Threshold {t:.1f} | Coverage: {coverage:5.1f}% "
        f"({n_covered}/{len(y_true)}) | Accuracy within coverage: {acc_at_t*100:.2f}%"
    )
    print(line)
    referral_lines.append(line)

with open(report_path, "a") as f:
    f.write("\n\nConfidence-Based Referral Breakdown:\n")
    f.write("\n".join(referral_lines))

# ===== PER-SAMPLE PREDICTIONS CSV =====
results_df = eval_df.copy()
results_df["predicted_label"] = [class_names[i] for i in y_pred]
results_df["confidence"] = y_confidence
results_df["correct"] = (y_true == y_pred)

results_path = os.path.join(BASE_DIR, f"evaluation_predictions_{RUN_TAG}.csv")
results_df.to_csv(results_path, index=False)
print(f"\nSaved per-sample predictions to: {results_path}")

# ===== AUTO-COMPARE AGAINST PREVIOUS RUNS =====
print("\n" + "="*60)
print("COMPARISON AGAINST PREVIOUS RUNS")
print("="*60)

previous_reports = [
    f for f in os.listdir(BASE_DIR)
    if re.match(r"evaluation_report.*\.txt$", f) and f != os.path.basename(report_path)
]

if not previous_reports:
    print("No previous evaluation_report*.txt files found to compare against.")
else:
    for fname in sorted(previous_reports):
        fpath = os.path.join(BASE_DIR, fname)
        with open(fpath, "r") as f:
            content = f.read()
        match = re.search(r"Overall Accuracy:\s*([\d.]+)%", content)
        if match:
            prev_acc = float(match.group(1))
            diff = overall_acc * 100 - prev_acc
            sign = "+" if diff >= 0 else ""
            print(f"  {fname}: {prev_acc:.2f}%  ->  this run: {overall_acc*100:.2f}%  ({sign}{diff:.2f} pts)")
        else:
            print(f"  {fname}: could not parse accuracy")