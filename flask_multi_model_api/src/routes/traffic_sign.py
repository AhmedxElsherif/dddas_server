import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import base64
import io
from PIL import Image
import sys
import os # Import os for path joining
from flask import Blueprint, request, jsonify

# Create a Blueprint
traffic_sign_bp = Blueprint("traffic_sign_bp", __name__)

# --- Global Variables (Load model and labels once) ---
MODEL_FILENAME = "model_trained.keras" # Original filename
LABEL_FILENAME = "labels.csv" # Original filename
# Construct path relative to the parent (src) directory
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(SRC_DIR, MODEL_FILENAME)
LABEL_PATH = os.path.join(SRC_DIR, LABEL_FILENAME)

model = None
label_data = None
threshold = 0.7 # Prediction threshold

def load_traffic_sign_resources():
    """Loads the Keras model and labels CSV."""
    global model, label_data
    if model is not None and label_data is not None:
        return True # Already loaded
        
    try:
        print(f"Attempting to load traffic sign model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Traffic sign model loaded successfully.")
        
        print(f"Attempting to load traffic sign labels from {LABEL_PATH}...")
        if not os.path.exists(LABEL_PATH):
             raise FileNotFoundError(f"Label file not found at {LABEL_PATH}")
            
        label_data = pd.read_csv(LABEL_PATH)
        print("Traffic sign labels loaded successfully.")
        return True
        
    except FileNotFoundError as e:
        print(f"Error loading traffic sign resources: {e}. Please ensure model and label files exist in the src directory.", 
              file=sys.stderr)
        model = None
        label_data = None
        return False
    except Exception as e:
        print(f"Error loading traffic sign resources: {str(e)}", file=sys.stderr)
        model = None
        label_data = None
        return False

# --- Helper Functions (Adapted from original script) ---
def getClassName(classNo):
    if label_data is None:
        return "Error: Labels not loaded"
    try:
        # Ensure classNo is within the bounds of the label data
        if 0 <= classNo < len(label_data):
            # Assuming the CSV has columns like ClassId and Name

            # Adjust column name if different
            return label_data.iloc[classNo]["Name"] 
        else:
            return "Error: Invalid class index"
    except KeyError:
         # Fallback if Name column doesnt exist, try index 1 (common for simple label files)

         if len(label_data.columns) > 1:
             return label_data.iloc[classNo][label_data.columns[1]]
         else:
             return "Error: Label format error"
    except Exception as e:
        print(f"Error getting class name for index {classNo}: {e}", file=sys.stderr)
        return "Error: Label lookup failed"

def preprocessing(img_rgb):
    """Preprocesses an RGB image for the traffic sign model."""
    try:
        # Resize first
        img_resized = cv2.resize(img_rgb, (32, 32))
        # Convert to Grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        # Equalize Histogram
        img_eq = cv2.equalizeHist(img_gray)
        # Normalize
        img_norm = img_eq / 255.0
        return img_norm
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        return None

def classify_traffic_sign_api(frame_rgb):
    """Classifies traffic sign from an RGB image frame."""
    if frame_rgb is None or frame_rgb.size == 0:
        return None, 0.0
    
    if model is None or label_data is None:
        print("Error: Traffic sign model or labels not initialized.", file=sys.stderr)
        if not load_traffic_sign_resources():
            return "Error: Model not ready", 0.0

    try:
        # Preprocess the image
        img_processed = preprocessing(frame_rgb)
        if img_processed is None:
             return "Error: Preprocessing failed", 0.0
             
        # Reshape for model input (1 image, height, width, channels)
        img_input = img_processed.reshape(1, 32, 32, 1)

        # Make prediction
        predictions = model.predict(img_input, verbose=0)
        classIndex = int(np.argmax(predictions)) # Ensure index is int
        probabilityValue = float(np.max(predictions)) # Ensure probability is float

        # Check threshold and return result
        if probabilityValue > threshold:
            className = getClassName(classIndex)
            return className, probabilityValue
        else:
            return "No sign detected", probabilityValue # Or return None, probabilityValue

    except Exception as e:
        print(f"Error processing image in classify_traffic_sign_api: {str(e)}", file=sys.stderr)
        return "Error: Classification failed", 0.0

# --- Flask Endpoint --- 
@traffic_sign_bp.route("/predict", methods=["POST"])
def predict_traffic_sign():
    """Receives an image frame (base64 encoded) and returns the prediction."""
    if model is None or label_data is None:
        if not load_traffic_sign_resources():
             return jsonify({"error": "Traffic sign model and resources could not be loaded."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "Missing image key in JSON payload"}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(data["image"])
        # Convert to numpy array using PIL/io
        image = Image.open(io.BytesIO(image_data))
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        frame_rgb = np.array(image)

        # Classify the traffic sign
        sign_name, confidence = classify_traffic_sign_api(frame_rgb)

        # Return the result
        return jsonify({
            "sign": sign_name,
            "confidence": confidence
        })

    except base64.binascii.Error:
        return jsonify({"error": "Invalid base64 string"}), 400
    except Exception as e:
        print(f"Error in /predict endpoint (traffic sign): {str(e)}", file=sys.stderr)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# Load resources eagerly when the app starts (using Flasks app context)
# load_traffic_sign_resources()

