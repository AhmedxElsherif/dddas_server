import tkinter as tk
from tkinter import ttk, messagebox
import requests
import threading
import time
import cv2
import pyaudio
import base64
import io
from PIL import Image, ImageTk
import numpy as np
import soundfile as sf

# Attempt to import Picamera2
USE_PICAMERA2 = False
try:
    from picamera2 import Picamera2
    # Import controls if you plan to use Enums like AfModeEnum, otherwise direct integers are fine.
    # from libcamera import controls as picam2_controls # Renamed to avoid conflict if user has other 'controls'
    USE_PICAMERA2 = True
    print("[INFO] Picamera2 library found. Will attempt to use it.")
except ImportError:
    print("[INFO] Picamera2 library not found. Will use OpenCV VideoCapture as fallback.")
    Picamera2 = None
    # picam2_controls = None

# --- Configuration ---
FLASK_API_URL = "http://127.0.0.1:5000"
SIGN_LANG_ENDPOINT = f"{FLASK_API_URL}/api/sign_language/predict"
TRAFFIC_SIGN_ENDPOINT = f"{FLASK_API_URL}/api/traffic_sign/predict"
SIREN_ENDPOINT = f"{FLASK_API_URL}/api/siren_detection/predict"

CV2_CAMERA_INDEX = 0 # OpenCV camera index if Picamera2 is not used

# Improved Camera Settings (to be used by Picamera2 primarily)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60 # Target FPS, actual might be lower based on processing
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT) # For OpenCV fallback and general reference

PICAM2_PREVIEW_CONFIG_NAME = "main" # For Picamera2, stream name for capture_array

AUDIO_RATE = 22050
AUDIO_CHUNK_SECONDS = 3
AUDIO_CHUNK_FRAMES = AUDIO_RATE * AUDIO_CHUNK_SECONDS
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1

UPDATE_INTERVAL_MS = 100 # Reduced for potentially faster UI updates with higher FPS

# --- Global Variables ---
root = None
camera_label = None
sign_lang_result_label = None
traffic_sign_result_label = None
siren_result_label = None

cv2_cap = None
picam2_instance = None

p_audio = None
audio_stream = None

stop_threads = False
last_sign_result = ("", 0.0)
last_traffic_result = ("", 0.0)
last_siren_result = (False, "None", 0.0)

# --- Helper Functions (encode_image_base64, encode_audio_base64 - Unchanged) ---
def encode_image_base64(frame_bgr_or_rgb, is_rgb=False):
    if frame_bgr_or_rgb is None or frame_bgr_or_rgb.size == 0:
        return None
    try:
        if is_rgb:
            frame_rgb = frame_bgr_or_rgb
        else:
            frame_rgb = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def encode_audio_base64(audio_data_np):
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data_np, AUDIO_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None

# --- API Call Functions (call_sign_language_api, call_traffic_sign_api, call_siren_api - Unchanged) ---
def call_sign_language_api(base64_image):
    global last_sign_result
    try:
        response = requests.post(SIGN_LANG_ENDPOINT, json={"image": base64_image}, timeout=2)
        response.raise_for_status()
        result = response.json()
        last_sign_result = (result.get("gesture", "Error"), result.get("confidence", 0.0))
    except requests.exceptions.RequestException as e:
        print(f"Sign Lang API Error: {e}")
        last_sign_result = ("API Error", 0.0)
    except Exception as e:
        print(f"Sign Lang Processing Error: {e}")
        last_sign_result = ("Processing Error", 0.0)

def call_traffic_sign_api(base64_image):
    global last_traffic_result
    try:
        response = requests.post(TRAFFIC_SIGN_ENDPOINT, json={"image": base64_image}, timeout=2)
        response.raise_for_status()
        result = response.json()
        last_traffic_result = (result.get("sign", "Error"), result.get("confidence", 0.0))
    except requests.exceptions.RequestException as e:
        print(f"Traffic Sign API Error: {e}")
        last_traffic_result = ("API Error", 0.0)
    except Exception as e:
        print(f"Traffic Sign Processing Error: {e}")
        last_traffic_result = ("Processing Error", 0.0)

def call_siren_api(base64_audio):
    global last_siren_result
    try:
        response = requests.post(SIREN_ENDPOINT, json={"audio": base64_audio}, timeout=4)
        response.raise_for_status()
        result = response.json()
        last_siren_result = (
            result.get("siren_detected", False),
            result.get("siren_type", "Error"),
            result.get("detection_confidence", 0.0)
        )
    except requests.exceptions.RequestException as e:
        print(f"Siren API Error: {e}")
        last_siren_result = (False, "API Error", 0.0)
    except Exception as e:
        print(f"Siren Processing Error: {e}")
        last_siren_result = (False, "Processing Error", 0.0)

# --- Update Functions ---
def update_camera_feed():
    global last_sign_result, last_traffic_result
    if stop_threads:
        return

    frame = None
    is_rgb_frame = False

    if USE_PICAMERA2 and picam2_instance:
        try:
            frame = picam2_instance.capture_array(PICAM2_PREVIEW_CONFIG_NAME) # Picamera2 gives RGB
            if frame is not None and frame.shape[-1] == 4: # Check for RGBA and convert to RGB if necessary
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            is_rgb_frame = True
        except Exception as e:
            print(f"Error capturing frame from Picamera2: {e}")
            frame = None
    elif cv2_cap and cv2_cap.isOpened():
        ret, cv2_frame = cv2_cap.read()
        if ret and cv2_frame is not None:
            frame = cv2_frame # OpenCV gives BGR
            is_rgb_frame = False
        else:
            print("Failed to capture frame from OpenCV camera.")
    else:
        if not stop_threads and root: root.after(UPDATE_INTERVAL_MS, update_camera_feed)
        return

    if frame is not None:
        # Display frame (resized for performance on UI)
        display_frame_resized = cv2.resize(frame, (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2))
        
        base64_image = encode_image_base64(frame, is_rgb=is_rgb_frame)
        
        if base64_image:
            threading.Thread(target=call_sign_language_api, args=(base64_image,), daemon=True).start()
            threading.Thread(target=call_traffic_sign_api, args=(base64_image,), daemon=True).start()

        sign_text = f"Sign: {last_sign_result[0]} ({last_sign_result[1]:.2f})"
        traffic_text = f"Traffic: {last_traffic_result[0]} ({last_traffic_result[1]:.2f})"
        if sign_lang_result_label: sign_lang_result_label.config(text=sign_text)
        if traffic_sign_result_label: traffic_sign_result_label.config(text=traffic_text)

        try:
            if not is_rgb_frame:
                img_display_rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
            else:
                img_display_rgb = display_frame_resized

            img_pil = Image.fromarray(img_display_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            if camera_label:
                camera_label.imgtk = img_tk
                camera_label.config(image=img_tk)
        except Exception as e:
            print(f"Error updating camera preview: {e}")

    if not stop_threads and root: 
        root.after(UPDATE_INTERVAL_MS, update_camera_feed)

def update_audio_feed(): # Unchanged
    global last_siren_result
    if stop_threads or audio_stream is None or not audio_stream.is_active():
        if not stop_threads and root: root.after(UPDATE_INTERVAL_MS, update_audio_feed)
        return
    try:
        data = audio_stream.read(AUDIO_CHUNK_FRAMES, exception_on_overflow=False)
        audio_data_np = np.frombuffer(data, dtype=np.int16)
        base64_audio = encode_audio_base64(audio_data_np)
        if base64_audio:
            threading.Thread(target=call_siren_api, args=(base64_audio,), daemon=True).start()
        siren_detected, siren_type, siren_conf = last_siren_result
        siren_text = f"Siren: {siren_type} ({siren_conf:.2f})" if siren_detected else f"Siren: None ({siren_conf:.2f})"
        if siren_result_label: siren_result_label.config(text=siren_text)
    except IOError as e:
        print(f"Audio read error: {e}")
    except Exception as e:
        print(f"Audio processing error: {e}")
    if not stop_threads and root: 
        root.after(UPDATE_INTERVAL_MS, update_audio_feed)

# --- Initialization and Teardown ---
def initialize_resources():
    global cv2_cap, picam2_instance, p_audio, audio_stream
    camera_initialized = False
    print("Initializing camera with improved settings...")

    if USE_PICAMERA2 and Picamera2 is not None:
        try:
            print("[INFO] Attempting to initialize with Picamera2...")
            picam2_instance = Picamera2()
            
            # Using create_video_configuration for more control, similar to the first script
            picam2_config = picam2_instance.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}, # Or XRGB8888 if preferred
                controls={
                    "FrameRate": CAMERA_FPS,
                    "AfMode": 2,  # Continuous autofocus (0=manual, 1=auto, 2=continuous)
                    "AfSpeed": 1, # Fast autofocus speed (0=normal, 1=fast)
                    # "AeEnable": True, # Auto Exposure, often default
                    # "AwbEnable": True, # Auto White Balance, often default
                }
            )
            print(f"[INFO] Picamera2 config created: {picam2_config}")
            picam2_instance.configure(picam2_config)
            
            # Explicitly set controls again if needed, though usually covered by create_video_configuration
            # picam2_instance.set_controls({"AfMode": 2, "AfSpeed": 1})
            
            picam2_instance.start()
            time.sleep(2) # Allow camera to initialize and autofocus to settle
            
            test_frame = picam2_instance.capture_array(PICAM2_PREVIEW_CONFIG_NAME) # Use the 'main' stream
            if test_frame is None or test_frame.size == 0:
                raise RuntimeError("Picamera2 started but failed to capture initial test frame.")
            print(f"[INFO] Picamera2 initialized successfully. Initial frame shape: {test_frame.shape}")
            camera_initialized = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Picamera2: {e}. Falling back to OpenCV if possible.")
            if picam2_instance: 
                try: picam2_instance.close() 
                except: pass # Ignore errors during close if already problematic
                picam2_instance = None

    if not camera_initialized and cv2:
        print("[INFO] Attempting to initialize with OpenCV VideoCapture...")
        cv2_cap = cv2.VideoCapture(CV2_CAMERA_INDEX)
        if not cv2_cap.isOpened():
            messagebox.showerror("Error", f"Cannot open OpenCV camera index {CV2_CAMERA_INDEX}")
            return False
        cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cv2_cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS) # Attempt to set FPS for OpenCV
        ret, frame = cv2_cap.read()
        if not ret or frame is None:
            messagebox.showerror("Error", f"Failed to read initial frame from OpenCV camera {CV2_CAMERA_INDEX}.")
            if cv2_cap: cv2_cap.release(); cv2_cap = None
            return False
        print(f"[INFO] OpenCV Camera {CV2_CAMERA_INDEX} opened. Resolution: {cv2_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cv2_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, FPS: {cv2_cap.get(cv2.CAP_PROP_FPS)}")
        camera_initialized = True
    
    if not camera_initialized:
        messagebox.showerror("Fatal Error", "Could not initialize any camera source (Picamera2 or OpenCV).")
        return False

    print("Initializing audio...")
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK_FRAMES)
        print("Audio stream opened.")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open audio stream: {e}")
        cleanup_resources()
        return False
        
    try:
        requests.get(FLASK_API_URL, timeout=2)
        print("Flask server connection successful.")
    except requests.exceptions.ConnectionError:
        messagebox.showwarning("Warning", f"Could not connect to Flask server at {FLASK_API_URL}.")
        
    return True

def cleanup_resources(): # Unchanged
    global stop_threads, cv2_cap, picam2_instance, audio_stream, p_audio
    print("Cleaning up resources...")
    stop_threads = True
    time.sleep(0.5) 
    if picam2_instance:
        try:
            picam2_instance.stop()
            picam2_instance.close()
            print("Picamera2 stopped and closed.")
        except Exception as e:
            print(f"Error closing Picamera2: {e}")
        picam2_instance = None
    if cv2_cap:
        cv2_cap.release()
        print("OpenCV Camera released.")
        cv2_cap = None
    if audio_stream:
        if audio_stream.is_active(): audio_stream.stop_stream()
        audio_stream.close()
        print("Audio stream closed.")
        audio_stream = None
    if p_audio:
        p_audio.terminate()
        print("PyAudio terminated.")
        p_audio = None
    if root:
        try:
            root.quit()
            # root.destroy() # destroy can cause issues if called from protocol handler sometimes
            print("Tkinter quit called.")
        except tk.TclError:
            print("Tkinter window already destroyed or not fully initialized.")
        except Exception as e:
            print(f"Error during Tkinter cleanup: {e}")

# --- GUI Setup (Unchanged) ---
def create_gui():
    global root, camera_label, sign_lang_result_label, traffic_sign_result_label, siren_result_label
    root = tk.Tk()
    root.title("Multi-Model AI Dashboard (Picam2 Improved)")
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    camera_label = ttk.Label(main_frame, text="Camera Feed Loading...")
    camera_label.grid(row=0, column=0, columnspan=2, pady=5)
    results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
    results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    sign_lang_result_label = ttk.Label(results_frame, text="Sign Language: Waiting...", font=("TkDefaultFont", 12))
    sign_lang_result_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    traffic_sign_result_label = ttk.Label(results_frame, text="Traffic Sign: Waiting...", font=("TkDefaultFont", 12))
    traffic_sign_result_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    siren_result_label = ttk.Label(results_frame, text="Siren: Waiting...", font=("TkDefaultFont", 12))
    siren_result_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    status_var = tk.StringVar()
    status_bar = ttk.Label(main_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    status_var.set("Initializing...")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    if initialize_resources():
        status_var.set("Running...")
        print("[DEBUG] Starting update_camera_feed and update_audio_feed loops.")
        update_camera_feed()
        update_audio_feed()
    else:
        status_var.set("Initialization Failed. Close the window.")
        # No need to call cleanup_resources here as initialize_resources should handle it on failure
        # or on_closing will handle it.
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, cleaning up...")
        on_closing() # Use the same closing logic

def on_closing():
    print("WM_DELETE_WINDOW or KeyboardInterrupt: Cleaning up and exiting...")
    cleanup_resources()
    if root:
        root.destroy() # Ensure root is destroyed after cleanup

# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    try:
        create_gui()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup is called if create_gui exits unexpectedly, though on_closing should handle most cases
        if not stop_threads: # If cleanup hasn't been called effectively
             cleanup_resources()
        print("Application finished.")

