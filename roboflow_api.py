from inference_sdk import InferenceHTTPClient

# Apni actual Roboflow API key yahan rakhein
ROBOFLOW_API_KEY = "Q6sC4Q5DbcsGgx88orut"
WORKSPACE_NAME = "nivrutti-kolamkar"
WORKFLOW_ID = "skin-lesion-classification-with-97.8-accuracy-im2or"

HIGH_RISK_CLASSES = {
    "Melanoma", "Squamous Cell Carcinoma", "Actinic Keratosis",
    "Dermatofibroma", "Atopic Dermatitis", "Benign Keratosis"
}
MIN_FLAG_CONFIDENCE = 0.01  # Threshold thoda aur flexible kar diya hai (20%)


def get_cancer_risk_screening(image_path):
    """
    Runs the image through Roboflow's dermoscopic lesion classifier and
    checks whether any high-risk (potentially malignant) class was flagged.
    """
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": image_path}
        )

        # Yeh terminal mein print karega ki server kya bhej raha hai
        print("DEBUG - Full Roboflow Response:", result)

        # Safely extract predictions based on common workflow response structures
        predictions = None
        if isinstance(result, list) and len(result) > 0:
            if "predictions" in result[0]:
                preds_data = result[0]["predictions"]
                if isinstance(preds_data, dict) and "predictions" in preds_data:
                    predictions = preds_data["predictions"]
                elif isinstance(preds_data, dict):
                    predictions = preds_data

        if not predictions:
            print("DEBUG - No predictions dictionary found in Roboflow response.")
            return {"flagged": False, "class_name": None, "confidence": 0.0}

        best_class = None
        best_confidence = 0.0

        # Handle predictions whether they come as list/dict formats
        if isinstance(predictions, dict):
            for class_name, info in predictions.items():
                conf = info.get("confidence", 0.0) if isinstance(
                    info, dict) else 0.0
                if class_name in HIGH_RISK_CLASSES and conf > best_confidence:
                    best_class = class_name
                    best_confidence = conf

        if best_class and best_confidence >= MIN_FLAG_CONFIDENCE:
            return {
                "flagged": True,
                "class_name": best_class,
                "confidence": round(best_confidence * 100, 2),
            }

        return {"flagged": False, "class_name": None, "confidence": 0.0}

    except Exception as exc:
        print(f"Roboflow cancer-risk screening failed: {exc}")
        return None
