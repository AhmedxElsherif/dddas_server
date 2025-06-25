import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import platform
import sys

# Import Picamera2 and controls (optional)
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAM2_AVAILABLE = True
    print("Picamera2 library found.")
except ImportError:
    PICAM2_AVAILABLE = False
    print("Picamera2 library not found. Will try to use OpenCV for webcam access.")

# --- Configuration ---
MODEL_PATH = "sign_language_model.h5"
CAMERA_RESOLUTION = (640, 480) # Width, Height
USE_PICAMERA = PICAM2_AVAILABLE # Set to False to force OpenCV even if picam2 is available
OPENCV_CAMERA_INDEX = 0
# --- End Configuration ---

# Load the TensorFlow model
def load_model(model_path):
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        # Optional: Print model summary
        # model.summary()
        return model
    except Exception as e:
        print(f"Error loading model ({model_path}): {str(e)}", file=sys.stderr)
        print("Please ensure the model file exists and is accessible.", file=sys.stderr)
        return None

model = load_model(MODEL_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 # Added for potentially better tracking
)
mp_drawing = mp.solutions.drawing_utils

# Define class names (ensure this matches the model's training)
sign_language_classes = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "ExcuseMe",
    "F", "Food", "G", "H", "Hello", "Help", "House", "I", "I Love You", "J", "K", "L",
    "M", "N", "No", "O", "P", "Please", "Q", "R", "S", "T", "ThankYou", "U", "V", "W",
    "X", "Y", "Yes", "Z"
]

def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def pad_landmarks():
    # Pad with zeros for the second hand if only one is detected
    # 21 landmarks * 3 coordinates (x, y, z)
    return [0.0] * (21 * 3)

def classify_gesture(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        print("Warning: Received empty frame for classification.", file=sys.stderr)
        return None, None, None
    
    if model is None:
        print("Error: Model not loaded. Cannot classify gesture.", file=sys.stderr)
        return None, None, None
        
    try:
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Optimize processing
        result = hands.process(image_rgb)
        image_rgb.flags.writeable = True # Re-enable writing if needed later
        
        gesture = None
        confidence = 0.0
        hand_landmarks_for_drawing = result.multi_hand_landmarks

        if result.multi_hand_landmarks:
            combined_landmarks = []
            
            # Process first hand
            combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[0]))
            
            # Process second hand or pad
            if len(result.multi_hand_landmarks) > 1:
                combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[1]))
            else:
                combined_landmarks.extend(pad_landmarks())
                
            # Make prediction
            landmarks_array = np.array(combined_landmarks, dtype=np.float32).reshape(1, -1)
            
            # Ensure the input shape matches the model's expected input shape
            # Assuming model expects (1, 126) for 2 hands * 21 landmarks * 3 coords
            expected_features = 126 
            if landmarks_array.shape[1] != expected_features:
                 print(f"Warning: Landmark data shape {landmarks_array.shape} != expected ({expected_features}). Padding/truncating.", file=sys.stderr)
                 if landmarks_array.shape[1] < expected_features:
                     landmarks_array = np.pad(landmarks_array, ((0,0), (0, expected_features - landmarks_array.shape[1])), 'constant')
                 else:
                     landmarks_array = landmarks_array[:, :expected_features]
            
            # Predict
            prediction = model.predict(landmarks_array, verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = prediction[0][class_id]
            
            # Get gesture name
            if 0 <= class_id < len(sign_language_classes):
                gesture = sign_language_classes[class_id]
            else:
                print(f"Error: Predicted class ID {class_id} is out of bounds.", file=sys.stderr)
                gesture = "Error"
        
        return gesture, hand_landmarks_for_drawing, confidence
        
    except Exception as e:
        print(f"Error processing image in classify_gesture: {str(e)}", file=sys.stderr)
        return None, None, None

def display_results(frame_bgr, gesture, hand_landmarks, confidence, fps):
    if frame_bgr is None:
        return
        
    try:
        # Draw landmarks
        if hand_landmarks:
            for landmarks in hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr, landmarks, mp_hands.HAND_CONNECTIONS)

        # Display prediction text
        if gesture:
            text = f"{gesture} ({confidence:.2%})"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "No sign detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display FPS
        cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow("Hand Sign Recognition", frame_bgr)

    except Exception as e:
        print(f"Error displaying results: {str(e)}", file=sys.stderr)

def run_camera_picam2(): 
    picam2 = None
    print("Attempting to start Picamera2...")
    try:
        picam2 = Picamera2()
        # Configure for video, specified resolution, RGB format needed by MediaPipe
        config = picam2.create_video_configuration(
            main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
            controls={
                "FrameRate": 30, # Request 30 FPS
                "AfMode": controls.AfModeEnum.Continuous, 
                "AwbEnable": True, 
                "AwbMode": controls.AwbModeEnum.Auto
            }
        )
        picam2.configure(config)
        print(f"Picamera2 configured: {config}")
        
        picam2.start()
        print("Picamera2 started successfully.")
        time.sleep(1) # Allow camera to initialize

    except Exception as e:
        print(f"Failed to initialize or start Picamera2: {str(e)}", file=sys.stderr)
        if picam2:
            try:
                picam2.stop()
            except Exception as stop_e:
                print(f"Error stopping Picamera2 during init failure: {stop_e}", file=sys.stderr)
        return

    prev_time = time.time()
    frame_count = 0
    fps = 0

    print("Starting camera loop (Picamera2). Press 'q' to quit.")
    while True:
        try:
            # Capture frame as RGB numpy array
            frame_rgb = picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV display/drawing consistency
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Flip horizontally for a mirror effect
            frame_bgr = cv2.flip(frame_bgr, 1)

            # --- Processing ---
            gesture, hand_landmarks, confidence = classify_gesture(frame_bgr.copy()) # Pass a copy if classify_gesture modifies it
            # --- End Processing ---

            # FPS calculation (more stable over time)
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                prev_time = current_time

            # Display results
            display_results(frame_bgr, gesture, hand_landmarks, confidence, fps)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed.")
                break

        except Exception as e:
            print(f"Error during Picamera2 loop: {str(e)}", file=sys.stderr)
            # Optional: break the loop on error, or try to continue
            # break 
            time.sleep(1) # Pause if error occurs rapidly

    # Cleanup
    print("Stopping Picamera2...")
    if picam2:
        try:
            picam2.stop()
            print("Picamera2 stopped.")
        except Exception as e:
            print(f"Error stopping Picamera2: {str(e)}", file=sys.stderr)
    cv2.destroyAllWindows()
    print("OpenCV windows closed.")

def run_camera_opencv():
    print(f"Attempting to open OpenCV camera index {OPENCV_CAMERA_INDEX}...")
    cap = cv2.VideoCapture(OPENCV_CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    if not cap.isOpened():
        print(f"Error: Could not open OpenCV webcam (index {OPENCV_CAMERA_INDEX}).", file=sys.stderr)
        return
        
    print(f"OpenCV Webcam {OPENCV_CAMERA_INDEX} opened successfully.")
    print(f"Resolution set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    prev_time = time.time()
    frame_count = 0
    fps = 0

    print("Starting camera loop (OpenCV). Press 'q' to quit.")
    while True:
        ret, frame_bgr = cap.read() # Reads as BGR
        if not ret:
            print("Error: Failed to read frame from OpenCV webcam. Exiting.", file=sys.stderr)
            break

        # Flip horizontally
        frame_bgr = cv2.flip(frame_bgr, 1)

        # --- Processing ---
        gesture, hand_landmarks, confidence = classify_gesture(frame_bgr.copy())
        # --- End Processing ---

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            prev_time = current_time

        # Display results
        display_results(frame_bgr, gesture, hand_landmarks, confidence, fps)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed.")
            break
            
    # Cleanup
    print("Releasing OpenCV camera...")
    cap.release()
    print("OpenCV camera released.")
    cv2.destroyAllWindows()
    print("OpenCV windows closed.")

if __name__ == "__main__":
    print("--- Hand Sign Recognition Script ---")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"MediaPipe version: {mp.__version__}")
    
    if model is None:
        print("Model could not be loaded. Please check the file path and integrity. Exiting.", file=sys.stderr)
        sys.exit(1)

    if USE_PICAMERA:
        run_camera_picam2()
    else:
        if not PICAM2_AVAILABLE:
             print("Picamera2 not available, using OpenCV.")
        else:
             print("USE_PICAMERA set to False, using OpenCV.")
        run_camera_opencv()
        
    print("Script finished.")

