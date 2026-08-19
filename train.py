import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from model import build_model, IMG_SIZE, NUM_CLASSES

BASE_DIR = r"C:\Users\VICTUS\Videos\Final_year_project"
TRAIN_CSV = os.path.join(BASE_DIR, "train_consolidated.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_consolidated.csv")
CLASS_LIST_FILE = os.path.join(BASE_DIR, "class_list_consolidated.txt")

BATCH_SIZE = 16
STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 20

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

with open(CLASS_LIST_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

num_classes = len(class_names)
print(f"Loaded {num_classes} consolidated classes.")

class_to_idx = {name: idx for idx, name in enumerate(class_names)}
train_df["label_idx"] = train_df["Disease_label"].map(class_to_idx)
val_df["label_idx"] = val_df["Disease_label"].map(class_to_idx)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label_idx"]),
    y=train_df["label_idx"].values
)
softened_weights = class_weights ** 0.5
class_weight_dict = dict(enumerate(softened_weights))

print("Softened Class Weights:")
for name, idx in class_to_idx.items():
    print(f"  {name}: {class_weight_dict[idx]:.3f}")

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

def load_and_preprocess(filepath, label, is_training=False):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    if is_training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    img = preprocess_input(img)
    return img, label

def create_dataset(df, is_training=False):
    filepaths = df["filepath"].values
    labels = df["label_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: load_and_preprocess(x, y, is_training=is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_df, is_training=True)
val_ds = create_dataset(val_df, is_training=False)

model = build_model(num_classes=num_classes)
if isinstance(model, tuple):
    model, base_model_ref = model

print("\n" + "="*60)
print("STAGE 1: Training classification head (EfficientNetB0 frozen)")
print("="*60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(BASE_DIR, "model_stage1_consolidated.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )
]

history_stage1 = model.fit(
    train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS,
    class_weight=class_weight_dict, callbacks=callbacks_stage1
)

print("\n" + "="*60)
print("STAGE 2: Fine-tuning top 60 layers")
print("="*60)

base_model = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(BASE_DIR, "model_stage2_consolidated.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )
]

history_stage2 = model.fit(
    train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
    class_weight=class_weight_dict, callbacks=callbacks_stage2
)

final_path = os.path.join(BASE_DIR, "final_model_consolidated.keras")
model.save(final_path)
print(f"\n✅ Retraining complete! Final model saved to: {final_path}")