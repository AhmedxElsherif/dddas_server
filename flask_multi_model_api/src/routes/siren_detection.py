import librosa
import numpy as np
import tensorflow as tf
from scipy import signal
import base64
import io
import soundfile as sf # To read audio data from bytes
import sys
import os # Import os for path joining
from flask import Blueprint, request, jsonify

# Create a Blueprint
siren_detection_bp = Blueprint("siren_detection_bp", __name__)

# --- Global Variables (Load models and constants once) ---
# Assume models are in the parent ( src ) directory
MODEL_DETECTION_FILENAME = "Detection.h5" # Original filename
MODEL_RECOGNITION_FILENAME = "Recognition_2.h5" # Original filename
# Construct path relative to the parent ( src ) directory
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DETECTION_PATH = os.path.join(SRC_DIR, MODEL_DETECTION_FILENAME)
MODEL_RECOGNITION_PATH = os.path.join(SRC_DIR, MODEL_RECOGNITION_FILENAME)

model_detection = None
model_recognition = None
sos = None
detection_input_shape = None
recognition_input_shape = None

RATE = 22050 # Expected sample rate
DETECTION_THRESHOLD = 0.5
SIREN_TYPES = {0: "Ambulance", 1: "Firetruck", 2: "Police", 3: "Traffic"}

def load_siren_resources():
    """Loads the siren detection and recognition models and filter."""
    global model_detection, model_recognition, sos, detection_input_shape, recognition_input_shape
    if model_detection is not None and model_recognition is not None and sos is not None:
        return True # Already loaded
        
    try:
        print(f"Attempting to load siren detection model from {MODEL_DETECTION_PATH}...")
        if not os.path.exists(MODEL_DETECTION_PATH):
             raise FileNotFoundError(f"Detection model file not found at {MODEL_DETECTION_PATH}")
        model_detection = tf.keras.models.load_model(MODEL_DETECTION_PATH)
        detection_input_shape = model_detection.input_shape[1:]
        print(f"Siren detection model loaded. Input shape: {detection_input_shape}")

        print(f"Attempting to load siren recognition model from {MODEL_RECOGNITION_PATH}...")
        if not os.path.exists(MODEL_RECOGNITION_PATH):
             raise FileNotFoundError(f"Recognition model file not found at {MODEL_RECOGNITION_PATH}")
        model_recognition = tf.keras.models.load_model(MODEL_RECOGNITION_PATH)
        # Assuming the first layer holds the input shape info
        recognition_input_shape = model_recognition.layers[0].input_shape[1:] 
        print(f"Siren recognition model loaded. Input shape: {recognition_input_shape}")

        print("Creating Butterworth filter...")
        sos = signal.butter(5, [50, 5000], "bandpass", fs=RATE, output="sos")
        print("Butterworth filter created.")
        return True
        
    except FileNotFoundError as e:
        print(f"Error loading siren resources: {e}. Please ensure model files exist in the src directory.",
            file=sys.stderr
        )
        model_detection = None
        model_recognition = None
        sos = None
        return False
    except Exception as e:
        print(f"Error loading siren resources: {str(e)}", file=sys.stderr)
        model_detection = None
        model_recognition = None
        sos = None
        return False

# --- Helper Functions (Adapted from original script) ---
def preprocess_detection_api(audio_data):
    """Preprocesses audio data for the detection model."""
    if sos is None or detection_input_shape is None:
        print("Error: Filter or detection model shape not initialized.", file=sys.stderr)
        return None
    try:
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
             audio_data = audio_data.astype(np.float32)
             # Normalize if it was integer type before (assuming standard int16 range)
             if np.max(np.abs(audio_data)) > 1.0: 
                 audio_data /= 32767.0
                 
        # Apply filter
        audio_data_filtered = signal.sosfilt(sos, audio_data)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data_filtered, sr=RATE, n_mfcc=40)
        # Pad MFCCs
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, max(0, detection_input_shape[1] - mfccs.shape[1]))), mode="constant")
        # Reshape and add batch dimension
        mfccs_reshaped = mfccs_padded.reshape(detection_input_shape)
        mfccs_batch = np.expand_dims(mfccs_reshaped, axis=0)
        return mfccs_batch
    except Exception as e:
        print(f"Error during detection preprocessing: {e}", file=sys.stderr)
        return None

def preprocess_recognition_api(audio_data):
    """Preprocesses audio data for the recognition model."""
    if recognition_input_shape is None:
        print("Error: Recognition model shape not initialized.", file=sys.stderr)
        return None
    try:
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
             audio_data = audio_data.astype(np.float32)
             if np.max(np.abs(audio_data)) > 1.0:
                 audio_data /= 32767.0
                 
        # Take the first second (as per original code)
        audio_segment = audio_data[:RATE] 
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=RATE, n_mfcc=80)
        # Scale features (mean across time)
        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        # Reshape and add batch dimension
        mfccs_reshaped = mfccs_scaled_features.reshape(recognition_input_shape)
        mfccs_batch = np.expand_dims(mfccs_reshaped, axis=0)
        return mfccs_batch
    except Exception as e:
        print(f"Error during recognition preprocessing: {e}", file=sys.stderr)
        return None

def classify_siren_api(audio_data):
    """Detects and recognizes siren from an audio data chunk."""
    if model_detection is None or model_recognition is None or sos is None:
        print("Error: Siren models or filter not initialized.", file=sys.stderr)
        if not load_siren_resources():
            return False, "Error: Models not ready", 0.0

    try:
        # --- Detection --- 
        detection_input = preprocess_detection_api(audio_data)
        if detection_input is None:
            return False, "Error: Detection preprocessing failed", 0.0
            
        predicted_proba_vector = model_detection.predict(detection_input, verbose=0)
        siren_prob = float(predicted_proba_vector[0][1]) # Probability of class 1 (siren)

        # --- Recognition (only if detected) --- 
        siren_type = "None"
        if siren_prob > DETECTION_THRESHOLD:
            recognition_input = preprocess_recognition_api(audio_data)
            if recognition_input is None:
                return True, "Error: Recognition preprocessing failed", siren_prob # Detected, but recognition failed

            recognition_prediction = model_recognition.predict(recognition_input, verbose=0)
            predicted_class_index = int(np.argmax(recognition_prediction[0]))
            
            if predicted_class_index in SIREN_TYPES:
                siren_type = SIREN_TYPES[predicted_class_index]
            else:
                print(f"Error: Predicted siren class index {predicted_class_index} is invalid.", file=sys.stderr)
                siren_type = "Error: Unknown class"
            
            return True, siren_type, siren_prob
        else:
            # No siren detected
            return False, "None", siren_prob

    except Exception as e:
        print(f"Error processing audio in classify_siren_api: {str(e)}", file=sys.stderr)
        return False, "Error: Classification failed", 0.0

# --- Flask Endpoint --- 
@siren_detection_bp.route("/predict", methods=["POST"])
def predict_siren():
    """Receives an audio chunk (base64 encoded WAV/FLAC/OGG) and returns detection/recognition."""
    if model_detection is None or model_recognition is None or sos is None:
        if not load_siren_resources():
             return jsonify({"error": "Siren models and resources could not be loaded."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "audio" not in data:
        return jsonify({"error": "Missing \\\\\\naudio\\\\\\n key in JSON payload (base64 encoded audio file)"}), 400

    try:
        # Decode the base64 audio
        audio_data_base64 = data["audio"]
        audio_bytes = base64.b64decode(audio_data_base64)
        
        # Read audio data from bytes using soundfile
        # soundfile automatically handles various formats (WAV, FLAC, OGG, etc.)
        # It also provides the sample rate, which we need to verify
        audio_array, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Check sample rate
        if samplerate != RATE:
            # Optional: Resample the audio if needed, or return an error
            # For now, return error if sample rate doesnt match models expected rate
            return jsonify({"error": f"Incorrect audio sample rate. Expected {RATE} Hz, got {samplerate} Hz."}), 400
            # Example resampling (requires librosa): 
            # audio_array = librosa.resample(y=audio_array.astype(np.float32), orig_sr=samplerate, target_sr=RATE)

        # If stereo, convert to mono (e.g., by averaging channels)
        if audio_array.ndim > 1 and audio_array.shape[1] > 1:
            audio_array = np.mean(audio_array, axis=1)
            
        # Classify the audio chunk
        detected, siren_type, confidence = classify_siren_api(audio_array)

        # Return the result
        return jsonify({
            "siren_detected": detected,
            "siren_type": siren_type,
            "detection_confidence": confidence # Confidence of *detection* model
        })

    except base64.binascii.Error:
        return jsonify({"error": "Invalid base64 string for audio data"}), 400
    except sf.SoundFileError as e:
         return jsonify({"error": f"Could not read audio data: {e}. Ensure its a supported format (WAV, FLAC, OGG)."}), 400
    except Exception as e:
        print(f"Error in /predict endpoint (siren detection): {str(e)}", file=sys.stderr)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# Load resources eagerly when the app starts (using Flasks app context)
# load_siren_resources()

