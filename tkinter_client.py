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

UPDATE_INTERVAL_MS = 100 # How often to update camera/audio feeds (milliseconds)

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
    try:
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG") # Or PNG
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def encode_audio_base64(audio_data_np):
    """Encodes a numpy audio array to base64 string (WAV format)."""
    try:
        buffer = io.BytesIO()
        # Save as WAV file in memory
        sf.write(buffer, audio_data_np, AUDIO_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None

# --- API Call Functions (Run in Threads) ---
def call_sign_language_api(base64_image):
    global last_sign_result
    try:
        response = requests.post(SIGN_LANG_ENDPOINT, json={"image": base64_image}, timeout=2)
        response.raise_for_status() # Raise an exception for bad status codes
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
        response = requests.post(SIREN_ENDPOINT, json={"audio": base64_audio}, timeout=4) # Longer timeout for audio
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
    if stop_threads or cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if ret:
        # Resize for display if needed
        display_frame = cv2.resize(frame, (CAMERA_RESOLUTION[0] // 2, CAMERA_RESOLUTION[1] // 2))
        
        # Encode and send to APIs in separate threads
        base64_image = encode_image_base64(frame)
        if base64_image:
            threading.Thread(target=call_sign_language_api, args=(base64_image,), daemon=True).start()
            threading.Thread(target=call_traffic_sign_api, args=(base64_image,), daemon=True).start()

        # Update GUI labels with the *last known* results
        sign_text = f"Sign: {last_sign_result[0]} ({last_sign_result[1]:.2f})"
        traffic_text = f"Traffic: {last_traffic_result[0]} ({last_traffic_result[1]:.2f})"
        sign_lang_result_label.config(text=sign_text)
        traffic_sign_result_label.config(text=traffic_text)

        # Update camera image in GUI
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        camera_label.imgtk = img_tk
        camera_label.config(image=img_tk)

    # Schedule next update
    root.after(UPDATE_INTERVAL_MS, update_camera_feed)

def update_audio_feed():
    global last_siren_result
    if stop_threads or audio_stream is None or not audio_stream.is_active():
        return

    try:
        # Read audio chunk
        data = audio_stream.read(AUDIO_CHUNK_FRAMES, exception_on_overflow=False)
        audio_data_np = np.frombuffer(data, dtype=np.int16)
        
        # Check if audio chunk is mostly silence (optional optimization)
        # rms = np.sqrt(np.mean(audio_data_np.astype(np.float32)**2))
        # if rms < 100: # Adjust threshold
        #     root.after(UPDATE_INTERVAL_MS, update_audio_feed)
        #     return
            
        # Encode and send to API in a separate thread
        base64_audio = encode_audio_base64(audio_data_np)
        if base64_audio:
            threading.Thread(target=call_siren_api, args=(base64_audio,), daemon=True).start()

        # Update GUI label with the *last known* result
        siren_detected, siren_type, siren_conf = last_siren_result
        siren_text = f"Siren: {siren_type} ({siren_conf:.2f})" if siren_detected else f"Siren: None ({siren_conf:.2f})"
        siren_result_label.config(text=siren_text)

    except IOError as e:
        print(f"Audio read error: {e}")
    except Exception as e:
        print(f"Audio processing error: {e}")

    # Schedule next update (adjust timing based on chunk size)
    # We process chunks of AUDIO_CHUNK_SECONDS, so waiting slightly less than that might be okay,
    # but calling too frequently might overload the API/processing.
    # Let's stick to UPDATE_INTERVAL_MS for simplicity, but be aware of implications.
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
    print(f"Camera {CAMERA_INDEX} opened.")

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
        
    # Check Flask server connection (optional but recommended)
    try:
        requests.get(FLASK_API_URL, timeout=2)
        print("Flask server connection successful.")
    except requests.exceptions.ConnectionError:
        messagebox.showwarning("Warning", f"Could not connect to Flask server at {FLASK_API_URL}. Ensure it is running.")
        # Allow continuing, but API calls will fail
        
    return True

def cleanup_resources():
    global stop_threads, cap, audio_stream, p
    print("Cleaning up resources...")
    stop_threads = True
    time.sleep(0.5) # Allow threads to potentially finish current task
    if cap:
        cap.release()
        print("Camera released.")
    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
        print("Audio stream closed.")
    if p:
        p.terminate()
        print("PyAudio terminated.")
    if root:
        root.quit()
        root.destroy()
        print("Tkinter window destroyed.")

# --- GUI Setup ---
def create_gui():
    global root, camera_label, sign_lang_result_label, traffic_sign_result_label, siren_result_label

    root = tk.Tk()
    root.title("Multi-Model AI Dashboard")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Camera Feed Label
    camera_label = ttk.Label(main_frame, text="Camera Feed Loading...")
    camera_label.grid(row=0, column=0, columnspan=2, pady=5)

    # Results Frame
    results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
    results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    sign_lang_result_label = ttk.Label(results_frame, text="Sign Language: Waiting...", font=("TkDefaultFont", 12))
    sign_lang_result_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

    traffic_sign_result_label = ttk.Label(results_frame, text="Traffic Sign: Waiting...", font=("TkDefaultFont", 12))
    traffic_sign_result_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)

    siren_result_label = ttk.Label(results_frame, text="Siren: Waiting...", font=("TkDefaultFont", 12))
    siren_result_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)

    # Status Bar (Optional)
    status_var = tk.StringVar()
    status_bar = ttk.Label(main_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    status_var.set("Initializing...")

    # Set cleanup on window close
    root.protocol("WM_DELETE_WINDOW", cleanup_resources)

    # Initialize resources and start updates
    if initialize_resources():
        status_var.set("Running...")
        update_camera_feed()
        update_audio_feed()
    else:
        status_var.set("Initialization Failed. Close the window.")

    root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    create_gui()

