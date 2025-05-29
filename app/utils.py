from lsmyolo import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import torch
import os

# Check if model file exists
MODEL_PATH = 'app/lsm_v9c.pt'
YAML_PATH = 'LSM-YOLO/LSM-YOLO.yaml'
model = None

def load_model():
    global model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model from YAML
        print(f"Initializing model from {YAML_PATH}")
        model = YOLO(YAML_PATH)
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading weights from {MODEL_PATH}")
            # Load weights directly
            weights = torch.load(MODEL_PATH, map_location=device)
            
            # Check what's in the weights file
            print(f"Keys in weights file: {weights.keys() if isinstance(weights, dict) else 'not a dict'}")
            
            # Try to load directly if weights is a state dict
            if not isinstance(weights, dict):
                print("Weights not in expected format, using as state dict directly")
                model.model.load_state_dict(weights)
            elif 'model' in weights:
                print("Loading 'model' from weights")
                model.model.load_state_dict(weights['model'])
            elif 'state_dict' in weights:
                print("Loading 'state_dict' from weights")
                model.model.load_state_dict(weights['state_dict'])
            else:
                print("No recognized model structure, attempting to load weights directly")
                model.model.load_state_dict(weights)
                
            model.to(device)
            print("Model loaded successfully and ready for inference")
        else:
            print(f"Error: Model file {MODEL_PATH} not found")
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
            
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise e

def predict_on_image(image_bytes: bytes):
    """
    Predict on image bytes
    """
    global model
    if model is None:
        try:
            load_model()
        except Exception as e:
            return {"error": f"Model not loaded: {str(e)}"}
    
    try:
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run inference
        results = model.predict(image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = round(float(box.conf[0]), 2)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "confidence": confidence
                })

        return {
            "detections": detections
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error during prediction: {str(e)}"}
        
def draw_detections(image_bytes: bytes, detections: list):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV dùng BGR
        #lật ảnh theo trục dọc
        # img_cv = cv2.flip(img_cv, 1)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']}: {det['confidence']}"
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        is_success, buffer = cv2.imencode(".jpg", img_cv)
        if not is_success:
            return None
        return buffer.tobytes()
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return None