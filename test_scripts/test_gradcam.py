import tensorflow as tf
import numpy as np
from gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_gradcam
from model import IMG_SIZE

MODEL_PATH = r"C:\Users\VICTUS\Videos\Final_year_project\final_model_consolidated.keras"
TEST_IMAGE = r"C:\Users\VICTUS\Videos\Final_year_project\DATASET\DATASET_0\DATASET_0\IMG_2075.jpg"  # apni koi test image ka path daalo
CLASS_LIST_FILE = r"C:\Users\VICTUS\Videos\Final_year_project\class_list_consolidated.txt"

with open(CLASS_LIST_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

model = tf.keras.models.load_model(MODEL_PATH)
base_model, last_conv_layer_name = find_last_conv_layer(model)
print(f"Using last conv layer: {last_conv_layer_name}")

# Load and preprocess image same as training
preprocess_input = tf.keras.applications.efficientnet.preprocess_input
img = tf.io.read_file(TEST_IMAGE)
img = tf.image.decode_image(img, channels=3, expand_animations=False)
img.set_shape([None, None, 3])
img = tf.image.resize(img, IMG_SIZE)
img = tf.cast(img, tf.float32)
img_preprocessed = preprocess_input(img)
img_array = tf.expand_dims(img_preprocessed, axis=0)

heatmap, pred_index, all_probs = make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name)

print(f"Predicted class: {class_names[pred_index]}")
print(f"Confidence: {all_probs[pred_index]*100:.2f}%")

overlay_gradcam(TEST_IMAGE, heatmap, save_path="gradcam_output.jpg")
print("Saved: gradcam_output.jpg")