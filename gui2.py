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
import os
import subprocess

# Attempt to import Picamera2
USE_PICAMERA2 = False
try:
    from picamera2 import Picamera2
    from libcamera import controls
    USE_PICAMERA2 = True
    print("[INFO] Picamera2 library found. Will attempt to use it.")
except ImportError:
    print("[INFO] Picamera2 library not found. Will use OpenCV as fallback.")
    Picamera2 = None
    controls = None

# Configuration
FLASK_API_URL = "http://127.0.0.1:5000"
CAMERA_RESOLUTION = (640, 480)
AUDIO_RATE = 22050
UPDATE_INTERVAL_MS = 200

# Global variables
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

def check_camera_access():
    """Check camera accessibility and release if needed"""
    try:
        # Try accessing camera device directly
        if os.path.exists('/dev/video0'):
            os.access('/dev/video0', os.R_OK | os.W_OK)
        return True
    except Exception as e:
        print(f"Camera access error: {e}")
        return False

def initialize_camera():
    """Initialize camera with multiple fallback options"""
    global cv2_cap, picam2_instance
    
    # Try Picamera2 first (for Raspberry Pi Camera)
    if USE_PICAMERA2 and Picamera2 is not None:
        try:
            picam2_instance = Picamera2()
            config = picam2_instance.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
            )
            picam2_instance.configure(config)
            
            # Enable auto controls if available
            if controls:
                try:
                    picam2_instance.set_controls({
                        "AfMode": controls.AfModeEnum.Continuous,
                        "AwbEnable": True,
                        "AwbMode": controls.AwbModeEnum.Auto
                    })
                except Exception as e:
                    print(f"Couldn't set camera controls: {e}")
            
            picam2_instance.start()
            time.sleep(2)  # Camera warm-up
            
            # Test capture
            test_frame = picam2_instance.capture_array("main")
            if test_frame is not None:
                print("Picamera2 initialized successfully")
                return True
                
        except Exception as e:
            print(f"Picamera2 init failed: {e}")
            if picam2_instance:
                picam2_instance.close()
            picam2_instance = None

    # Fallback to OpenCV with multiple device tries
    video_devices = [
        0,  # Default camera index
        1,  # Secondary index
        '/dev/video0',
        '/dev/video1',
        '/dev/video10',
        '/dev/video12'
    ]
    
    for device in video_devices:
        try:
            cv2_cap = cv2.VideoCapture(device)
            if cv2_cap.isOpened():
                cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                
                # Test frame capture
                ret, frame = cv2_cap.read()
                if ret:
                    print(f"OpenCV camera initialized on device: {device}")
                    return True
                else:
                    cv2_cap.release()
        except Exception as e:
            print(f"OpenCV init failed on device {device}: {e}")
            if cv2_cap:
                cv2_cap.release()
    
    return False

def initialize_resources():
    """Initialize all system resources"""
    # First check camera access
    if not check_camera_access():
        messagebox.showwarning("Warning", 
            "Camera access denied. Try running with sudo or check permissions.")
        return False
    
    # Initialize camera
    if not initialize_camera():
        messagebox.showerror("Error",
            "Could not initialize camera.\n\n"
            "Possible solutions:\n"
            "1. Make sure camera is connected\n"
            "2. Enable camera in raspi-config\n"
            "3. Try different camera device\n"
            "4. Check if another app is using the camera")
        return False
    
    # Initialize audio
    try:
        global p_audio, audio_stream
        p_audio = pyaudio.PyAudio()
        audio_stream = p_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_RATE * 3  # 3 second chunks
        )
        print("Audio initialized successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Audio initialization failed: {e}")
        return False
    
    return True

def update_camera_feed():
    """Update the camera feed in GUI"""
    global stop_threads
    
    if stop_threads:
        return

    frame = None
    is_rgb = False

    try:
        if USE_PICAMERA2 and picam2_instance:
            frame = picam2_instance.capture_array("main")
            if frame is not None:
                if frame.shape[-1] == 4:  # RGBA to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                is_rgb = True
        elif cv2_cap and cv2_cap.isOpened():
            ret, frame = cv2_cap.read()
            if not ret:
                print("Failed to read frame from OpenCV")
                frame = None
    except Exception as e:
        print(f"Camera capture error: {e}")

    if frame is not None:
        try:
            # Display processing
            display_frame = cv2.resize(frame, (CAMERA_RESOLUTION[0]//2, CAMERA_RESOLUTION[1]//2))
            
            if not is_rgb:  # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            if camera_label:
                camera_label.imgtk = imgtk
                camera_label.config(image=imgtk)
                
            # API calls would go here
            # (Removed for brevity, add your API call logic back)
            
        except Exception as e:
            print(f"Frame processing error: {e}")

    if not stop_threads and root:
        root.after(UPDATE_INTERVAL_MS, update_camera_feed)

def cleanup_resources():
    """Clean up all resources"""
    global stop_threads
    
    print("Cleaning up resources...")
    stop_threads = True
    
    # Clean camera
    if picam2_instance:
        try:
            picam2_instance.stop()
            picam2_instance.close()
        except Exception as e:
            print(f"Error stopping Picamera2: {e}")
    
    if cv2_cap:
        try:
            cv2_cap.release()
        except Exception as e:
            print(f"Error releasing OpenCV camera: {e}")
    
    # Clean audio
    if audio_stream:
        try:
            audio_stream.stop_stream()
            audio_stream.close()
        except Exception as e:
            print(f"Error closing audio stream: {e}")
    
    if p_audio:
        try:
            p_audio.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
    
    if root:
        try:
            root.quit()
        except:
            pass

def create_gui():
    """Create the main application GUI"""
    global root, camera_label, sign_lang_result_label, traffic_sign_result_label, siren_result_label
    
    root = tk.Tk()
    root.title("AI Dashboard")
    
    # Main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Camera display
    camera_label = ttk.Label(main_frame, text="Initializing Camera...")
    camera_label.grid(row=0, column=0, columnspan=2, pady=5)
    
    # Results frame
    results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
    results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    # Result labels
    sign_lang_result_label = ttk.Label(results_frame, text="Sign Language: -", font=("TkDefaultFont", 10))
    sign_lang_result_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    
    traffic_sign_result_label = ttk.Label(results_frame, text="Traffic Sign: -", font=("TkDefaultFont", 10))
    traffic_sign_result_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    
    siren_result_label = ttk.Label(results_frame, text="Siren: -", font=("TkDefaultFont", 10))
    siren_result_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    
    # Status bar
    status_var = tk.StringVar(value="Initializing...")
    status_bar = ttk.Label(main_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    # Set close handler
    root.protocol("WM_DELETE_WINDOW", cleanup_resources)
    
    # Initialize resources
    if initialize_resources():
        status_var.set("Running")
        update_camera_feed()
        # Start other update functions here
    else:
        status_var.set("Initialization Failed")
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"GUI error: {e}")
        cleanup_resources()

if __name__ == "__main__":
    try:
        create_gui()
    except Exception as e:
        print(f"Fatal error: {e}")
        cleanup_resources()