#!/usr/bin/python3.11
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
    # from libcamera import controls # For autofocus, AWB
    USE_PICAMERA2 = True
    print("[INFO] Picamera2 library found. Will attempt to use it.")
except ImportError:
    print("[INFO] Picamera2 library not found. Will use OpenCV VideoCapture as fallback.")
    Picamera2 = None # Define for type hinting or checks if needed
    controls = None

# --- Configuration ---
FLASK_API_URL = "http://127.0.0.1:5000"
SIGN_LANG_ENDPOINT = f"{FLASK_API_URL}/api/sign_language/predict"
TRAFFIC_SIGN_ENDPOINT = f"{FLASK_API_URL}/api/traffic_sign/predict"
SIREN_ENDPOINT = f"{FLASK_API_URL}/api/siren_detection/predict"

CV2_CAMERA_INDEX = 0 # OpenCV camera index if Picamera2 is not used
CAMERA_RESOLUTION = (640, 480) # For Picamera2 and OpenCV
PICAM2_PREVIEW_CONFIG_NAME = "main" # For Picamera2, stream name for capture_array

AUDIO_RATE = 22050
AUDIO_CHUNK_SECONDS = 3
AUDIO_CHUNK_FRAMES = AUDIO_RATE * AUDIO_CHUNK_SECONDS
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1

UPDATE_INTERVAL_MS = 300

# --- Global Variables ---
root = None
camera_label = None
sign_lang_result_label = None
traffic_sign_result_label = None
siren_result_label = None

# Camera specific globals
cv2_cap = None
picam2_instance = None

p_audio = None
audio_stream = None

stop_threads = False
last_sign_result = ("", 0.0)
last_traffic_result = ("", 0.0)
last_siren_result = (False, "None", 0.0)

# --- Helper Functions ---
def encode_image_base64(frame_bgr_or_rgb, is_rgb=False):
    print("[DEBUG] encode_image_base64: Starting encoding...")
    if frame_bgr_or_rgb is None or frame_bgr_or_rgb.size == 0:
        print("[DEBUG] encode_image_base64: Frame is empty or None.")
        return None
    try:
        print(f"[DEBUG] encode_image_base64: Frame shape: {frame_bgr_or_rgb.shape}, dtype: {frame_bgr_or_rgb.dtype}")
        if is_rgb:
            frame_rgb = frame_bgr_or_rgb
        else: # Assume BGR
            frame_rgb = cv2.cvtColor(frame_bgr_or_rgb, cv2.COLOR_BGR2RGB)
        
        print("[DEBUG] encode_image_base64: Frame is RGB for PIL.")
        pil_image = Image.fromarray(frame_rgb)
        print("[DEBUG] encode_image_base64: Created PIL Image.")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        print("[DEBUG] encode_image_base64: Saved to buffer as JPEG.")
        encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("[DEBUG] encode_image_base64: Encoded to base64 string.")
        return encoded_string
    except Exception as e:
        print(f"[DEBUG] encode_image_base64: Error encoding image: {e}")
        import traceback
        traceback.print_exc()
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

# --- API Call Functions (Unchanged) ---
def call_sign_language_api(base64_image):
    global last_sign_result
    print("[DEBUG] call_sign_language_api: Sending request...")
    try:
        response = requests.post(SIGN_LANG_ENDPOINT, json={"image": base64_image}, timeout=2)
        response.raise_for_status()
        result = response.json()
        last_sign_result = (result.get("gesture", "Error"), result.get("confidence", 0.0))
        print(f"[DEBUG] call_sign_language_api: Received: {last_sign_result}")
    except requests.exceptions.RequestException as e:
        print(f"Sign Lang API Error: {e}")
        last_sign_result = ("API Error", 0.0)
    except Exception as e:
        print(f"Sign Lang Processing Error: {e}")
        last_sign_result = ("Processing Error", 0.0)

def call_traffic_sign_api(base64_image):
    global last_traffic_result
    print("[DEBUG] call_traffic_sign_api: Sending request...")
    try:
        response = requests.post(TRAFFIC_SIGN_ENDPOINT, json={"image": base64_image}, timeout=2)
        response.raise_for_status()
        result = response.json()
        last_traffic_result = (result.get("sign", "Error"), result.get("confidence", 0.0))
        print(f"[DEBUG] call_traffic_sign_api: Received: {last_traffic_result}")
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
def enhance_colors(frame_rgb):
    """
    Enhances the colors of an RGB frame by increasing saturation, brightness, and contrast.
    Args:
        frame_rgb: Input frame in RGB format (numpy array).
    Returns:
        Enhanced frame in RGB format.
    """
    try:
        # Step 1: Increase Brightness and Contrast
        alpha = 1.3  # Contrast control (1.0-3.0)
        beta = 10    # Brightness control (0-100)
        frame_adjusted = cv2.convertScaleAbs(frame_rgb, alpha=alpha, beta=beta)

        # Step 2: Increase Saturation in HSV space
        frame_hsv = cv2.cvtColor(frame_adjusted, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(frame_hsv)
        # Increase saturation by a factor (e.g., 1.5), but cap at 255
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        frame_hsv = cv2.merge([h, s, v])
        frame_enhanced = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)

        # Step 3: Optional - Simple White Balance
        # Split channels
        r, g, b = cv2.split(frame_enhanced)
        # Compute means
        r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
        # Compute scaling factors (normalize to green channel)
        if g_avg != 0:  # Avoid division by zero
            r_scale = g_avg / (r_avg + 1e-10)
            b_scale = g_avg / (b_avg + 1e-10)
            r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
            b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
        frame_enhanced = cv2.merge([r, g, b])

        return frame_enhanced
    except Exception as e:
        print(f"[ERROR] Color enhancement failed: {e}")
        return frame_rgb  # Return original frame if enhancement fails

# Update the update_camera_feed function to apply color enhancement
def update_camera_feed():
    global last_sign_result, last_traffic_result
    if stop_threads:
        return

    frame = None
    is_rgb_frame = False

    if USE_PICAMERA2 and picam2_instance:
        try:
            frame = picam2_instance.capture_array(PICAM2_PREVIEW_CONFIG_NAME)
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            is_rgb_frame = True
        except Exception as e:
            print(f"[DEBUG] Error capturing frame from Picamera2: {e}")
            frame = None
    elif cv2_cap and cv2_cap.isOpened():
        ret, cv2_frame = cv2_cap.read()
        if ret and cv2_frame is not None:
            frame = cv2_frame
            is_rgb_frame = False
        else:
            print("[DEBUG] update_camera_feed: Failed to capture frame from OpenCV camera.")
    else:
        print("[DEBUG] update_camera_feed: No camera source available.")
        if not stop_threads and root:
            root.after(UPDATE_INTERVAL_MS, update_camera_feed)
        return

    if frame is not None:
        # Apply color enhancement
        if is_rgb_frame:
            frame = enhance_colors(frame)
        else:
            # Convert BGR to RGB, enhance, then convert back if using OpenCV camera
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = enhance_colors(frame_rgb)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        display_frame_resized = cv2.resize(frame, (CAMERA_RESOLUTION[0] // 4, CAMERA_RESOLUTION[1] // 4))
        base64_image = encode_image_base64(frame, is_rgb=is_rgb_frame)
        if base64_image:
            threading.Thread(target=call_sign_language_api, args=(base64_image,), daemon=True).start()
            threading.Thread(target=call_traffic_sign_api, args=(base64_image,), daemon=True).start()
        else:
            print("[DEBUG] update_camera_feed: Failed to encode image.")

        sign_text = f"Sign: {last_sign_result[0]} ({last_sign_result[1]:.2f})"
        traffic_text = f"Traffic: {last_traffic_result[0]} ({last_traffic_result[1]:.2f})"
        if "Error" in last_traffic_result[0]:
            traffic_text = f"Traffic: Error - Check Server"
        if sign_lang_result_label:
            sign_lang_result_label.config(text=sign_text)
        if traffic_sign_result_label:
            traffic_sign_result_label.config(text=traffic_text)

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
            print(f"[DEBUG] update_camera_feed: Error updating camera preview: {e}")
    if not stop_threads and root:
        root.after(UPDATE_INTERVAL_MS, update_camera_feed)
def update_audio_feed(): # Unchanged from previous debug version
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
    print("Initializing camera...")

    if USE_PICAMERA2 and Picamera2 is not None:
        try:
            print("[INFO] Attempting to initialize with Picamera2...")
            picam2_instance = Picamera2()
            config = picam2_instance.create_video_configuration(
                main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
                controls={
                    "FrameRate": 30,
                    "AfMode": 2,
                    "AfSpeed": 1
                }
            )
            picam2_instance.configure(config)
            picam2_instance.start()
            time.sleep(2)
            test_frame = picam2_instance.capture_array(PICAM2_PREVIEW_CONFIG_NAME)
            if test_frame is None or test_frame.size == 0:
                raise RuntimeError("Picamera2 started but failed to capture initial test frame.")
            print(f"[INFO] Picamera2 initialized successfully. Initial frame shape: {test_frame.shape}")
            camera_initialized = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Picamera2: {e}")
            if picam2_instance:
                picam2_instance.close()
                picam2_instance = None
            print("[INFO] Skipping OpenCV fallback since Picamera2 is expected to work.")

    if not camera_initialized and not USE_PICAMERA2:
        print("[INFO] Attempting to initialize with OpenCV VideoCapture...")
        for index in range(4):
            print(f"[INFO] Trying camera index {index}...")
            cv2_cap = cv2.VideoCapture(index)
            if not cv2_cap.isOpened():
                print(f"[INFO] Cannot open camera index {index}")
                cv2_cap.release()
                continue
            cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
            cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
            time.sleep(1)
            ret, frame = cv2_cap.read()
            if ret and frame is not None:
                print(f"[INFO] OpenCV Camera {index} opened and initial frame read successfully. Frame shape: {frame.shape}")
                camera_initialized = True
                break
            print(f"[INFO] Failed to read initial frame from camera index {index}")
            cv2_cap.release()
        if not camera_initialized:
            print("[ERROR] Could not find any working OpenCV camera.")

    print("Initializing audio...")
    audio_initialized = False
    p_audio = pyaudio.PyAudio()
    try:
        # Enumerate audio devices for debugging
        print("[INFO] Enumerating audio devices...")
        for i in range(p_audio.get_device_count()):
            dev = p_audio.get_device_info_by_index(i)
            print(f"[INFO] Device {i}: {dev['name']}, Input Channels: {dev['maxInputChannels']}, Output Channels: {dev['maxOutputChannels']}")
        # Try opening the default input device
        audio_stream = p_audio.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_FRAMES
        )
        print("[INFO] Audio stream opened successfully.")
        audio_initialized = True
    except Exception as e:
        print(f"[ERROR] Cannot open audio stream: {e}")
        # Try to find a specific input device
        for i in range(p_audio.get_device_count()):
            dev = p_audio.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                try:
                    audio_stream = p_audio.open(
                        format=AUDIO_FORMAT,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,
                        input=True,
                        frames_per_buffer=AUDIO_CHUNK_FRAMES,
                        input_device_index=i
                    )
                    print(f"[INFO] Audio stream opened successfully on device {i}: {dev['name']}")
                    audio_initialized = True
                    break
                except Exception as e2:
                    print(f"[ERROR] Failed to open audio stream on device {i}: {e2}")
        if not audio_initialized:
            print("[WARNING] No audio input device found. Proceeding without audio.")

    try:
        response = requests.get(FLASK_API_URL, timeout=2)
        print(f"[INFO] Flask server connection successful: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[WARNING] Could not connect to Flask server at {FLASK_API_URL}")

    return camera_initialized, audio_initialized


def cleanup_resources():
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
            root.destroy()
            print("Tkinter window destroyed.")
        except tk.TclError:
            print("Tkinter window already destroyed or not fully initialized.")
        except Exception as e:
            print(f"Error destroying Tkinter window: {e}")

# --- GUI Setup (Unchanged) ---
def create_gui():
    global root, camera_label, sign_lang_result_label, traffic_sign_result_label, siren_result_label
    root = tk.Tk()
    root.title("Multi-Model AI Dashboard (Picam2 Attempt)")
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
    root.protocol("WM_DELETE_WINDOW", cleanup_resources)
    camera_ok, audio_ok = initialize_resources()
    if camera_ok and audio_ok:
        status_var.set("Running...")
        print("[DEBUG] Starting update_camera_feed and update_audio_feed loops.")
        update_camera_feed()
        update_audio_feed()
    elif camera_ok:
        status_var.set("Running (Camera Only, Audio Failed)...")
        print("[DEBUG] Audio failed, starting update_camera_feed only.")
        update_camera_feed()
        # Update siren label to indicate audio is unavailable
        if siren_result_label:
            siren_result_label.config(text="Siren: Audio Unavailable")
    else:
        status_var.set("Initialization Failed. Close the window.")
        print("[ERROR] Both camera and audio initialization failed.")
        cleanup_resources()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, cleaning up...")
        cleanup_resources()
# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    try:
        create_gui()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
        cleanup_resources()

