def initialize_resources():
    global cv2_cap, picam2_instance, p_audio, audio_stream
    camera_initialized = False
    print("Initializing camera...")

    if USE_PICAMERA2 and Picamera2 is not None:
        try:
            print("[INFO] Attempting to initialize with Picamera2...")
            picam2_instance = Picamera2()
            config = picam2_instance.create_preview_configuration(
                main={"format": "RGB888", "size": CAMERA_RESOLUTION},
                lores={"format": "YUV420", "size": (CAMERA_RESOLUTION[0]//2, CAMERA_RESOLUTION[1]//2)},
                display=None
            )
            picam2_instance.configure(config)
            picam2_instance.start()
            time.sleep(1)
            test_frame = picam2_instance.capture_array(PICAM2_PREVIEW_CONFIG_NAME)
            if test_frame is None or test_frame.size == 0:
                raise RuntimeError("Picamera2 started but failed to capture initial test frame.")
            print(f"[INFO] Picamera2 initialized successfully. Initial frame shape: {test_frame.shape}")
            camera_initialized = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Picamera2: {e}. Falling back to OpenCV if possible.")
            if picam2_instance: picam2_instance.close(); picam2_instance = None

    if not camera_initialized and cv2:
        print("[INFO] Attempting to initialize with OpenCV VideoCapture...")
        cv2_cap = cv2.VideoCapture(CV2_CAMERA_INDEX)
        if not cv2_cap.isOpened():
            print(f"Cannot open OpenCV camera index {CV2_CAMERA_INDEX}")
            return False, False  # Camera failed, audio not attempted
        cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        ret, frame = cv2_cap.read()
        if not ret or frame is None:
            print(f"Failed to read initial frame from OpenCV camera {CV2_CAMERA_INDEX}.")
            cv2_cap.release(); cv2_cap = None
            return False, False
        print(f"[INFO] OpenCV Camera {CV2_CAMERA_INDEX} opened and initial frame read successfully.")
        camera_initialized = True

    print("Initializing audio...")
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK_FRAMES)
        print("Audio stream opened.")
        return camera_initialized, True
    except Exception as e:
        print(f"Cannot open audio stream: {e}")
        cleanup_resources()
        return camera_initialized, False