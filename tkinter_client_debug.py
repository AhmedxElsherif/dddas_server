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
import soundfile as sf # For saving audio if needed, or handling chunks

# --- Configuration ---
FLASK_API_URL = "http://127.0.0.1:5000" # Replace with Raspberry Pi's IP if running GUI on another machine
SIGN_LANG_ENDPOINT = f"{FLASK_API_URL}/api/sign_language/predict"
TRAFFIC_SIGN_ENDPOINT = f"{FLASK_API_URL}/api/traffic_sign/predict"
SIREN_ENDPOINT = f"{FLASK_API_URL}/api/siren_detection/predict"

CAMERA_INDEX = 0 # Adjust if needed
CAMERA_RESOLUTION = (640, 480)

AUDIO_RATE = 22050
AUDIO_CHUNK_SECONDS = 3 # Must match Flask server expectation if processing chunks
AUDIO_CHUNK_FRAMES = AUDIO_RATE * AUDIO_CHUNK_SECONDS
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1

UPDATE_INTERVAL_MS = 200 # How often to update camera/audio feeds (milliseconds), increased for debugging

# --- Global Variables ---
root = None
camera_label = None
sign_lang_result_label = None
traffic_sign_result_label = None
siren_result_label = None

cap = None
p = None
audio_stream = None

stop_threads = False
last_sign_result = ("", 0.0)
last_traffic_result = ("", 0.0)
last_siren_result = (False, "None", 0.0)

# --- Helper Functions ---
def encode_image_base64(frame_bgr):
    """Encodes a BGR OpenCV frame to base64 string."""
    print("[DEBUG] encode_image_base64: Starting encoding...")
    if frame_bgr is None or frame_bgr.size == 0:
        print("[DEBUG] encode_image_base64: Frame is empty or None.")
        return None
    try:
        print(f"[DEBUG] encode_image_base64: Frame shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        print("[DEBUG] encode_image_base64: Converted to RGB.")
        pil_image = Image.fromarray(frame_rgb)
        print("[DEBUG] encode_image_base64: Created PIL Image.")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG") # Or PNG
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
    """Encodes a numpy audio array to base64 string (WAV format)."""
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data_np, AUDIO_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None

# --- API Call Functions (Run in Threads) ---
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
def update_camera_feed():
    global last_sign_result, last_traffic_result
    # print("[DEBUG] update_camera_feed: Tick") # Can be too verbose
    if stop_threads or cap is None or not cap.isOpened():
        print("[DEBUG] update_camera_feed: Camera not ready or stop_threads is True.")
        if not stop_threads: root.after(UPDATE_INTERVAL_MS, update_camera_feed) # Reschedule if not stopping
        return

    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"[DEBUG] update_camera_feed: Frame captured, shape: {frame.shape}")
        # Resize for display if needed
        display_frame = cv2.resize(frame, (CAMERA_RESOLUTION[0] // 2, CAMERA_RESOLUTION[1] // 2))
        
        print("[DEBUG] update_camera_feed: Encoding image for APIs...")
        base64_image = encode_image_base64(frame) # Use original full-res frame for APIs
        
        if base64_image:
            print("[DEBUG] update_camera_feed: Image encoded, starting API call threads.")
            threading.Thread(target=call_sign_language_api, args=(base64_image,), daemon=True).start()
            threading.Thread(target=call_traffic_sign_api, args=(base64_image,), daemon=True).start()
        else:
            print("[DEBUG] update_camera_feed: Failed to encode image.")

        # Update GUI labels with the *last known* results
        sign_text = f"Sign: {last_sign_result[0]} ({last_sign_result[1]:.2f})"
        traffic_text = f"Traffic: {last_traffic_result[0]} ({last_traffic_result[1]:.2f})"
        if sign_lang_result_label: sign_lang_result_label.config(text=sign_text)
        if traffic_sign_result_label: traffic_sign_result_label.config(text=traffic_text)

        # Update camera image in GUI
        try:
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            if camera_label:
                camera_label.imgtk = img_tk
                camera_label.config(image=img_tk)
            # print("[DEBUG] update_camera_feed: Camera preview updated.")
        except Exception as e:
            print(f"[DEBUG] update_camera_feed: Error updating camera preview: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[DEBUG] update_camera_feed: Failed to capture frame from camera.")

    # Schedule next update
    if not stop_threads and root: 
        root.after(UPDATE_INTERVAL_MS, update_camera_feed)

def update_audio_feed():
    global last_siren_result
    if stop_threads or audio_stream is None or not audio_stream.is_active():
        if not stop_threads: root.after(UPDATE_INTERVAL_MS, update_audio_feed) # Reschedule if not stopping
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
    global cap, p, audio_stream
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open camera index {CAMERA_INDEX}")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    # Test read one frame to ensure camera is working
    ret, frame = cap.read()
    if not ret or frame is None:
        messagebox.showerror("Error", f"Failed to read initial frame from camera {CAMERA_INDEX}.")
        cap.release()
        return False
    print(f"Camera {CAMERA_INDEX} opened and initial frame read successfully.")

    print("Initializing audio...")
    p = pyaudio.PyAudio()
    try:
        audio_stream = p.open(format=AUDIO_FORMAT,
                              channels=AUDIO_CHANNELS,
                              rate=AUDIO_RATE,
                              input=True,
                              frames_per_buffer=AUDIO_CHUNK_FRAMES)
        print("Audio stream opened.")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open audio stream: {e}")
        if cap: cap.release()
        if p: p.terminate()
        return False
        
    try:
        requests.get(FLASK_API_URL, timeout=2)
        print("Flask server connection successful.")
    except requests.exceptions.ConnectionError:
        messagebox.showwarning("Warning", f"Could not connect to Flask server at {FLASK_API_URL}. Ensure it is running.")
        
    return True

def cleanup_resources():
    global stop_threads, cap, audio_stream, p
    print("Cleaning up resources...")
    stop_threads = True
    time.sleep(0.5) 
    if cap:
        cap.release()
        print("Camera released.")
    if audio_stream:
        if audio_stream.is_active():
            audio_stream.stop_stream()
        audio_stream.close()
        print("Audio stream closed.")
    if p:
        p.terminate()
        print("PyAudio terminated.")
    if root:
        try:
            root.quit()
            root.destroy()
            print("Tkinter window destroyed.")
        except tk.TclError:
            print("Tkinter window already destroyed or not fully initialized.")

# --- GUI Setup ---
def create_gui():
    global root, camera_label, sign_lang_result_label, traffic_sign_result_label, siren_result_label

    root = tk.Tk()
    root.title("Multi-Model AI Dashboard")

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

    if initialize_resources():
        status_var.set("Running...")
        print("[DEBUG] Starting update_camera_feed and update_audio_feed loops.")
        update_camera_feed()
        update_audio_feed()
    else:
        status_var.set("Initialization Failed. Close the window.")
        # Attempt to clean up even if initialization failed partially
        cleanup_resources() 

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, cleaning up...")
        cleanup_resources()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        create_gui()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
        # Attempt cleanup if GUI creation fails catastrophically
        cleanup_resources()

