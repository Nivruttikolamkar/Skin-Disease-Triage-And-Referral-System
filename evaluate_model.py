
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Model Path
MODEL_PATH = r"C:\Users\VICTUS\Videos\Final_year_project\training_artifacts\final_model_consolidated.keras"# print("Loading model...")
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Test dataset folder path
TEST_DIR = os.path.join("data", "test")
IMG_SIZE = (224, 224)  # Agar size alag ho toh change kar lena

# Agar training ke waqt aapne class names ki koi specific list use ki thi, 
# toh wahi exact list yahan likhein (Alphabetical order: sorted(os.listdir(TEST_DIR)))
class_names = sorted(os.listdir(TEST_DIR))
print(f"Target Classes: {class_names}")

total_images = 0
correct_predictions = 0

print("\nStarting evaluation on test images...")
print("-" * 50)

for true_class_name in class_names:
    class_folder = os.path.join(TEST_DIR, true_class_name)
    
    if not os.path.isdir(class_folder):
        continue
        
    img_names = os.listdir(class_folder)
    for img_name in img_names:
        img_path = os.path.join(class_folder, img_name)
        
        try:
            # Image load & preprocess
            img = image.load_img(img_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalization
            
            # Prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_idx]
            
            total_images += 1
            if predicted_class_name == true_class_name:
                correct_predictions += 1
                
        except Exception as e:
            continue

print("-" * 50)
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print(f"Total Test Images Evaluated: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Model Test Accuracy: {accuracy:.2f}%")
else:
    print("No test images found!")