import numpy as np
import tensorflow as tf
import cv2

from model import IMG_SIZE

def find_last_conv_layer(model):
    """
    Finds the EfficientNet base model inside your full model,
    then finds its last convolutional layer by name.
    """
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find EfficientNet base layer in model.")

    # Find the last Conv2D layer inside the base model
    last_conv_layer_name = None
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name

    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found inside base model.")

    return base_model, last_conv_layer_name


def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed image, shape (1, 224, 224, 3)
    Returns a 2D heatmap (values 0-1) same aspect ratio as the conv layer output.
    """
    # Build a model that maps input image -> (last conv layer output, final predictions)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)
        # Pass base_model output through the rest of your model's layers
        # (GlobalAveragePooling -> Dropout -> Dense -> Dropout -> Dense)
        x = base_output
        for layer in model.layers:
            if layer.name == base_model.name:
                continue  # skip, already applied
        # Re-run the head manually using the model's own layer objects
        head_input = base_output
        found_base = False
        preds = None
        x = head_input
        for layer in model.layers:
            if layer is base_model:
                found_base = True
                continue
            if not found_base:
                continue
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
        preds = x

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), preds.numpy()[0]


def overlay_gradcam(original_img_path, heatmap, alpha=0.4, save_path=None):
    """
    original_img_path: path to the original image file
    heatmap: 2D numpy array from make_gradcam_heatmap
    Returns the overlaid image (BGR, as OpenCV format) and optionally saves it.
    """
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlaid = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)

    if save_path:
        cv2.imwrite(save_path, overlaid)

    return overlaid