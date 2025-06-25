import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image
import sys
import os # Import os for path joining
from flask import Blueprint, request, jsonify

# Create a Blueprint
sign_language_bp = Blueprint("sign_language_bp", __name__)

# --- Global Variables (Load models and initializers once) ---
# Assume model is in the 'src' directory or a subdirectory
MODEL_FILENAME = "sign_language_model.h5"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME) # Path relative to this routes file

model = None
mp_hands = None
hands = None
sign_language_classes = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "ExcuseMe",
    "F", "Food", "G", "H", "Hello", "Help", "House", "I", "I Love You", "J", "K", "L",
    "M", "N", "No", "O", "P", "Please", "Q", "R", "S", "T", "ThankYou", "U", "V", "W",
    "X", "Y", "Yes", "Z"
]

def load_sign_language_resources():
    """Loads the TensorFlow model and initializes MediaPipe Hands."""
    global model, mp_hands, hands
    if model is not None and hands is not None:
        return True # Already loaded
    try:
        print(f"Loading sign language model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
             print(f"Error: Model file not found at {MODEL_PATH}", file=sys.stderr)
             # Try loading from parent directory as fallback if routes are nested
             alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), MODEL_FILENAME)
             if os.path.exists(alt_path):
                 print(f"Trying alternative path: {alt_path}")
                 model = tf.keras.models.load_model(alt_path)
             else:
                 raise FileNotFoundError(f"Model file not found at {MODEL_PATH} or {alt_path}")
        else:
            model = tf.keras.models.load_model(MODEL_PATH)
        print("Sign language model loaded successfully.")
        
        print("Initializing MediaPipe Hands...")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False, # Process video stream
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        print("MediaPipe Hands initialized.")
        return True
    except Exception as e:
        print(f"Error loading sign language resources: {str(e)}", file=sys.stderr)
        model = None
        hands = None
        return False

# --- Helper Functions (Adapted from original script) ---
def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def pad_landmarks():
    return [0.0] * (21 * 3)

def classify_gesture_api(frame_bgr):
    """Classifies gesture from a BGR image frame."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None, 0.0 # Return None gesture, 0 confidence
    
    if model is None or hands is None:
        print("Error: Sign language model or MediaPipe not initialized.", file=sys.stderr)
        # Attempt to load them if not loaded
        if not load_sign_language_resources():
             return "Error: Model not ready", 0.0
        
    try:
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        print(f"[DEBUG] Input image shape: {image_rgb.shape}")
        image_rgb.flags.writeable = False
        result = hands.process(image_rgb)
        print(f"[DEBUG] Multi-hand landmarks detected: {bool(result.multi_hand_landmarks)}")
        # image_rgb.flags.writeable = True # Not needed if not drawing
        
        gesture = None
        confidence = 0.0

        if result.multi_hand_landmarks:
            combined_landmarks = []
            combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[0]))
            if len(result.multi_hand_landmarks) > 1:
                combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[1]))
            else:
                combined_landmarks.extend(pad_landmarks())
                
            landmarks_array = np.array(combined_landmarks, dtype=np.float32).reshape(1, -1)
            expected_features = 126
            if landmarks_array.shape[1] != expected_features:
                 print(f"Warning: Landmark data shape {landmarks_array.shape} != expected ({expected_features}). Padding/truncating.", file=sys.stderr)
                 if landmarks_array.shape[1] < expected_features:
                     landmarks_array = np.pad(landmarks_array, ((0,0), (0, expected_features - landmarks_array.shape[1])), 'constant')
                 else:
                     landmarks_array = landmarks_array[:, :expected_features]
            
            prediction = model.predict(landmarks_array, verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = float(prediction[0][class_id]) # Ensure confidence is JSON serializable
            
            if 0 <= class_id < len(sign_language_classes):
                gesture = sign_language_classes[class_id]
            else:
                print(f"Error: Predicted class ID {class_id} is out of bounds.", file=sys.stderr)
                gesture = "Error: Prediction index out of bounds"
        else:
            gesture = "No hand detected"
            confidence = 0.0

        return gesture, confidence
        
    except Exception as e:
        print(f"Error processing image in classify_gesture_api: {str(e)}", file=sys.stderr)
        return "Error: Processing failed", 0.0

# --- Flask Endpoint --- 
@sign_language_bp.route("/predict", methods=["POST"])
def predict_sign():
    """Receives an image frame (base64 encoded) and returns the prediction."""
    # Ensure resources are loaded
    if model is None or hands is None:
        if not load_sign_language_resources():
             return jsonify({"error": "Model and resources could not be loaded."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "Missing 'image' key in JSON payload"}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(data["image"])
        # Convert to numpy array using PIL/io
        image = Image.open(io.BytesIO(image_data))
        frame_rgb = np.array(image)
        # Convert RGB (from PIL) to BGR (for consistency if needed, though classify expects BGR)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Classify the gesture
        gesture, confidence = classify_gesture_api(frame_bgr)

        # Return the result
        return jsonify({
            "gesture": gesture,
            "confidence": confidence
        })

    except base64.binascii.Error:
        return jsonify({"error": "Invalid base64 string"}), 400
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}", file=sys.stderr)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# Load resources eagerly when the app starts (using Flask's app context)
# This requires modifying main.py slightly or using @sign_language_bp.before_app_first_request
# For simplicity, lazy loading in the endpoint is kept, but eager loading is better for performance.
# load_sign_language_resources()

