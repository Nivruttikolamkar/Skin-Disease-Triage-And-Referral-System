from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

API_KEY = "gSpsGywcLDJE9hjSDFS5"  # Apni real Roboflow API key yahan daalo
WORKSPACE_NAME = "nivrutti-kolamkar"
WORKFLOW_ID = "skin-lesion-classification-with-97.8-accuracy-im2or"

# Yahan 'uploads' folder mein jo image actually padi hai, uska naam likhein (jaise sample.jpg ya koi hash naam)
IMAGE_PATH = os.path.join("uploads", r"C:\Users\VICTUS\Videos\Final_year_project\static\uploads\d41782c9_images_(5).jpeg")

def test_workflow():
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )
    
    print("Sending image to Roboflow Workflow...")
    
    # Image open karein
    img = Image.open(IMAGE_PATH)
    
    # Workflow run karein
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={
            "image": img
        }
    )
    
    print("\n--- Workflow Response ---")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_workflow()