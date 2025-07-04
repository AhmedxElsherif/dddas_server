import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
import threading
import cv2
from picamera2 import Picamera2
from libcamera import Transform, controls

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognizer - Raspberry Pi 5")
        self.running = False
        self.picam2 = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # إعدادات الكاميرا المحسنة
        self.camera_config = {
            "size": (640, 480),
            "format": "XBGR8888",
            "transform": Transform(hflip=True, vflip=False),
            "controls": {
                "AwbEnable": True,
                "AwbMode": controls.AwbModeEnum.Auto,
                "ExposureTime": 16000,
                "AnalogueGain": 1.8,
                "Saturation": 1.8,
                "Contrast": 1,
                "FrameRate": 30
            }
        }

        self.initialize_model()
        self.setup_ui()

    def initialize_model(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        
        MODEL_PATH = "sign_language_model_52_Word_TF_2.15.0.h5"
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {os.path.abspath(MODEL_PATH)}")
            exit(1)

        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.mp_draw = mp.solutions.drawing_utils

    def setup_ui(self):
        self.label = Label(self.root)
        self.label.pack()

        self.pred_label = Label(self.root, text="", font=("Arial", 16))
        self.pred_label.pack()

        Button(self.root, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        Button(self.root, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        Button(self.root, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=5)

    def start_camera(self):
        if not self.running:
            try:
                self.running = True
                self.picam2 = Picamera2()
                
                config = self.picam2.create_preview_configuration(
                    main={"size": self.camera_config["size"], 
                         "format": self.camera_config["format"]},
                    transform=self.camera_config["transform"],
                    controls=self.camera_config["controls"]
                )
                
                self.picam2.configure(config)
                self.picam2.start()
                
                # Thread for frame capture
                threading.Thread(target=self.capture_frames, daemon=True).start()
                # Thread for processing
                threading.Thread(target=self.process_frames, daemon=True).start()
                
            except Exception as e:
                print(f"Camera error: {e}")
                self.stop_camera()

    def stop_camera(self):
        self.running = False
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

    def capture_frames(self):
        """Thread for continuous frame capture without processing"""
        while self.running and self.picam2:
            try:
                frame = self.picam2.capture_array()
                with self.frame_lock:
                    self.latest_frame = frame
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.01)

    def process_frames(self):
        """Thread for frame processing"""
        while self.running:
            try:
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.latest_frame.copy()
                
                # Convert and process frame
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply color correction
                frame_bgr = self.adjust_colors(frame_bgr)
                
                # Process gesture
                gesture, hand_landmarks, confidence = self.classify_gesture(frame_bgr)
                
                # Draw landmarks if detected
                if hand_landmarks:
                    self.draw_landmarks(frame_bgr, hand_landmarks)
                
                # Update display
                self.update_display(frame_bgr, gesture, confidence)
                
                time.sleep(0.02)  # Reduce CPU load
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)

    def adjust_colors(self, frame):
        """Improved color adjustment"""
        # Convert to HSV for better color control
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Adjust saturation and value
        s = cv2.multiply(s, 1.3)
        v = cv2.multiply(v, 1.1)
        
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return frame

    def classify_gesture(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        if result.multi_hand_landmarks:
            combined_landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                combined_landmarks += [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            
            # Pad landmarks if only one hand detected
            if len(result.multi_hand_landmarks) == 1:
                combined_landmarks += [0.0] * 63
                
            landmarks_array = np.array(combined_landmarks).reshape(1, -1)
            prediction = self.model.predict(landmarks_array, verbose=0)[0][:12]
            class_id = np.argmax(prediction)
            confidence = prediction[class_id]
            return sign_language_classes[class_id], result.multi_hand_landmarks, confidence

        return None, None, None

    def draw_landmarks(self, frame, hand_landmarks):
        for landmarks in hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    def update_display(self, frame, gesture, confidence):
        """Update display with processed frame"""
        if gesture:
            cv2.putText(frame, f"{gesture} ({confidence:.2%})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.pred_label.config(text=f"Gesture: {gesture}, Confidence: {confidence:.2%}")
        else:
            cv2.putText(frame, "No sign detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.pred_label.config(text="No sign detected")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            print("Failed to load image")
            return

        gesture, hand_landmarks, confidence = self.classify_gesture(frame)
        if hand_landmarks:
            self.draw_landmarks(frame, hand_landmarks)

        self.update_display(frame, gesture, confidence)

if __name__ == "__main__":
    sign_language_classes = [
        "Excuseme", "Far", "Fare", "Hello",
        "Here", "Left", "Near", "Right",
        "Taxi", "Thanks", "There", "Where"
    ]
    
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()