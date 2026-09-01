import json
from unittest.mock import patch
from PIL import Image

# Import the necessary modules from the backend
from app.api.routes_query import process_query
from app.schemas.query import QueryRequest
from app.vlm.model import VLM

validation_report = {
    "P4 payload construction": "FAIL",
    "4 distinct images": "FAIL",
    "Image dimensions": "FAIL",
    "Image content": "FAIL",
    "P2 evidence included": "FAIL",
    "Original P2 imagery preserved": "FAIL",
}

original_image_to_data_url = VLM.image_to_data_url

def intercept_generate(self, image=None, question="", evidence=None, images=None):
    print("intercept_generate called!")
    
    validation_report["P4 payload construction"] = "PASS"
    
    if evidence and "NDBI for urban changed" in evidence:
        validation_report["P2 evidence included"] = "PASS"
        
    if images and len(images) == 4:
        if "before" in images and "after" in images and "change_map" in images and "s1_composite" in images:
            validation_report["4 distinct images"] = "PASS"
            
    # Verify original imagery size (should be larger than 512)
    original_sizes_ok = True
    for key, img in images.items():
        if img.width <= 512 and img.height <= 512:
            original_sizes_ok = False
            break
            
    if original_sizes_ok:
        validation_report["Original P2 imagery preserved"] = "PASS"
        
    # Now simulate the payload construction to check sizes
    payload_sizes_ok = True
    content_ok = True
    
    for key, img in images.items():
        # Call the actual resizing function we wrote
        data_url = original_image_to_data_url(img)
        
        # Verify it's a JPEG
        if not data_url.startswith("data:image/jpeg;base64,"):
            content_ok = False
            
        # Verify it's non-empty
        if len(data_url) < 100:
            content_ok = False
            
        # Now we need to decode and check the actual size of the generated JPEG
        import base64
        import io
        encoded_data = data_url.split(",")[1]
        decoded = base64.b64decode(encoded_data)
        decoded_img = Image.open(io.BytesIO(decoded))
        
        if decoded_img.width > 512 or decoded_img.height > 512:
            payload_sizes_ok = False
            
    if payload_sizes_ok:
        validation_report["Image dimensions"] = "PASS"
    
    if content_ok:
        validation_report["Image content"] = "PASS"
        
    # Return a dummy string to bypass 402 Payment Required
    return "MOCKED VLM RESPONSE"


if __name__ == "__main__":
    req = QueryRequest(
        query="Compare urban/built-up change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]",
        aoi={
            "type": "Polygon",
            "coordinates": [[[151.195, -33.885], [151.225, -33.885], [151.225, -33.855], [151.195, -33.855], [151.195, -33.885]]]
        }
    )
    
    with patch.object(VLM, 'generate', new=intercept_generate):
        try:
            print("Executing query locally...")
            res = process_query(req)
            print("Query finished!")
        except Exception as e:
            print(f"Exception: {e}")
            
    print("\n--- VALIDATION REPORT ---")
    for k, v in validation_report.items():
        print(f"{k}: {v}")
    print("Actual VLM inference: BLOCKED BY PROVIDER 402")
