import torch
from lsmyolo import YOLO
import os

MODEL_PATH = 'app/lsm_v9c.pt'

def test_model_load():
    try:
        print(f"Checking if model file exists: {os.path.exists(MODEL_PATH)}")
        print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
        
        # Try to load model
        print("Attempting to load model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Try loading with verbose output
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully, type: {type(model)}")
        
        # Print model attributes
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):
                try:
                    value = getattr(model, attr)
                    print(f"  {attr}: {type(value)}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_load() 