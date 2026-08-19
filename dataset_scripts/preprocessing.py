import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ===== CONFIG =====
BASE = r"C:\Users\VICTUS\Videos\Final_year_project"
PREPROC_DIR = BASE   # <-- everything read/saved directly in root now

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# ===== LOAD CSVs FROM PHASE 2 =====
train_df = pd.read_csv(os.path.join(PREPROC_DIR, "train_final.csv"))
val_df = pd.read_csv(os.path.join(PREPROC_DIR, "val_final.csv"))
test_df = pd.read_csv(os.path.join(PREPROC_DIR, "test_final.csv"))

with open(os.path.join(PREPROC_DIR, "class_list.txt")) as f:
    class_names = [line.strip() for line in f.readlines()]

class_to_idx = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(class_names)

train_df["label_idx"] = train_df["Disease_label"].map(class_to_idx)
val_df["label_idx"] = val_df["Disease_label"].map(class_to_idx)
test_df["label_idx"] = test_df["Disease_label"].map(class_to_idx)


# ===== CLASS WEIGHTS =====
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_df["label_idx"].values
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print("\nClass weights (higher = rarer class, model penalized more for getting it wrong):")
for idx, w in class_weight_dict.items():
    print(f"  {class_names[idx]}: {w:.3f}")


# ===== IMAGE LOADING FUNCTION =====
def load_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


# ===== AUGMENTATION (train set only) =====

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.05),
    tf.keras.layers.RandomBrightness(0.05),
])
preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def augment_and_preprocess(img, label):
    img = data_augmentation(img, training=True)
    img = preprocess_input(img)
    return img, label


def preprocess_only(img, label):
    img = preprocess_input(img)
    return img, label


# ===== BUILD tf.data DATASETS =====
def build_dataset(df, training=False):
    filepaths = df["filepath"].values
    labels = df["label_idx"].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
        ds = ds.map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = build_dataset(train_df, training=True)
val_ds = build_dataset(val_df, training=False)
test_ds = build_dataset(test_df, training=False)

print("\nDatasets built successfully.")
print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")


# ===== SANITY CHECK: visualize a few augmented training images =====
def save_sample_batch():
    images, labels = next(iter(train_ds))
    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        img_display = (images[i].numpy() + 1) / 2.0
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title(class_names[labels[i].numpy()], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    save_path = os.path.join(PREPROC_DIR, "sample_augmented_batch.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved sample augmented batch to: {save_path}")


if __name__ == "__main__":
    save_sample_batch()

    with open(os.path.join(PREPROC_DIR, "class_weights.json"), "w") as f:
        json.dump({str(k): v for k, v in class_weight_dict.items()}, f, indent=2)
    print("Saved class_weights.json")

    print("\n✅ Phase 3 complete — data pipeline is ready for training.")