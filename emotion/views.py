import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from cvzone import putTextRect
import json
from django.template import loader
import os
import plotly.graph_objects as go
from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
import time

FRAME_INTERVAL = 5  # Process every 5th frame
frame_count = 0
last_processed_time = time.time()

def process_frame(frame):
    # Your existing frame processing code (e.g., emotion detection) goes here.
    pass

def capture_video_frame():
    global frame_count
    # Capture the video frame from the camera (you may be using OpenCV or another library)
    # For example:
    # frame = camera.read()

    # Check if it is the right frame to process (every FRAME_INTERVAL frames)
    if frame_count % FRAME_INTERVAL == 0:
        process_frame(frame)
    
    frame_count += 1

    # Optional: Delay the next frame capture to control the frame rate.
    current_time = time.time()
    elapsed_time = current_time - last_processed_time
    
    # Limit processing to every 5 seconds (or based on your desired time interval)
    if elapsed_time < 1 / 30:  # Assuming 30 FPS, adjust as needed.
        time.sleep(0.05)  # Sleep for a little to avoid maxing out CPU.
    
    last_processed_time = time.time()


DATA_FILE_PATH = 'emotion_data.json'

# Load the trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Define emotion labels
emotion_labels = ['sad', 'happy', 'angry', 'neutral', 'demotivated']

# Initialize emotion counts if the file does not exist
if not os.path.exists(DATA_FILE_PATH):
    with open(DATA_FILE_PATH, 'w') as file:
        json.dump({'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'demotivated': 0}, file)

def video_feed():
    cap = cv2.VideoCapture(0)
    angry_detected = False  # Track angry emotion for the alert

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Bounding box processing
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (105, 105, 105), 2)

                # Extract and preprocess face for model
                face = frame[y:y + h, x:x + w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)

                # Predict emotion
                predictions = model.predict(face)
                emotion = emotion_labels[np.argmax(predictions)]

                # Check for angry emotion
                if emotion == "angry" and not angry_detected:
                    angry_detected = True  # Trigger alert

                # Update emotion counts
                try:
                    with open(DATA_FILE_PATH, 'r') as file:
                        emotion_counts = json.load(file)
                except (json.JSONDecodeError, IOError):
                    emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'demotivated': 0}

                emotion_counts[emotion] += 1

                with open(DATA_FILE_PATH, 'w') as file:
                    json.dump(emotion_counts, file)

                # Display emotion on frame
                putTextRect(frame, emotion, (x, y - 10), colorR=(105, 105, 105))

        # Encode frame as JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Stream frame
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()
    # Pass 'angry_detected' to template
    return render(request, 'emotion_detection.html', {'angry_detected': angry_detected})

# Django view to get emotion data
def get_emotion_data(request):
    try:
        with open(DATA_FILE_PATH, 'r') as file:
            emotion_counts = json.load(file)
    except (json.JSONDecodeError, IOError):
        emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'demotivated': 0}
    return JsonResponse(emotion_counts)


# Django view for webcam feed
def webcam_feed(request):
    try:
        return StreamingHttpResponse(video_feed(),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return HttpResponse(f"Error: {str(e)}", status=500)

# Render the webcam page
def detect_emotion(request):
    return render(request, 'webcam_feed.html')

def generate_graph(request):
    # Read data from the JSON file
    try:
        with open(DATA_FILE_PATH, 'r') as file:
            data = json.load(file)
    except (json.JSONDecodeError, IOError):
        data = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'demotivated': 0}

    emotions = list(data.keys())
    counts = list(data.values())

    # Create 2D Plotly figure
    fig_2d = go.Figure(data=[go.Bar(
        x=emotions,
        y=counts,
        name='Emotion Counts'
    )])

    fig_2d.update_layout(
        title='2D Emotion Counts',
        xaxis_title='Emotion',
        yaxis_title='Count'
    )

    # Create 3D Plotly figure
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=emotions,  # X-axis
        y=[0] * len(emotions),  # Dummy data for Y-axis
        z=counts,  # Z-axis
        mode='markers+lines',
        marker=dict(size=8),
        line=dict(width=2)
    )])

    fig_3d.update_layout(
        title='3D Emotion Counts',
        scene=dict(
            xaxis_title='Emotion',
            yaxis_title='Dummy',
            zaxis_title='Count'
        )
    )

    # Convert the figures to HTML
    graph_html_2d = fig_2d.to_html(full_html=False)
    graph_html_3d = fig_3d.to_html(full_html=False)

    # Render the HTML template with the graphs
    template = loader.get_template('graph.html')
    context = {
        'graph_html_2d': graph_html_2d,
        'graph_html_3d': graph_html_3d
    }
    return HttpResponse(template.render(context, request))

from django.shortcuts import render, redirect
import json
def detect(request):
    angry_count = 0  # Initialize the count for "angry" emotion

    # Read the emotion data file
    try:
        with open(DATA_FILE_PATH, 'r') as file:
            emotion_counts = json.load(file)
        # Get the "angry" count if it exists
        angry_count = emotion_counts.get('angry', 0)
    except (json.JSONDecodeError, IOError):
        # Handle error if the file is missing or empty
        emotion_counts = {}

    # Pass the "angry" count to the template
    return render(request, 'alert.html', {'angry_count': angry_count})

from django.shortcuts import render

def alert(request):
    # Render the alert.html template
    return render(request, 'alert.html')

def multilingual(request):
    return render(request, 'speaker.html')

from django.shortcuts import render
import random
import string
import hashlib
from cryptography.fernet import Fernet

def generate_key_from_seed(seed: str) -> bytes:
    # Convert seed to a 32-byte key using SHA-256 hashing
    sha256 = hashlib.sha256()
    sha256.update(seed.encode())
    return Fernet.generate_key()

def encrypt_data(data: str, key: bytes) -> bytes:
    # Encrypt data
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data: bytes, key: bytes) -> str:
    # Decrypt data
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data)
    return decrypted_data.decode()

import os

def geocite_interface(request):
    seed = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    key = generate_key_from_seed(seed)

    # Example of data you want to encrypt
    sensitive_data = "This is a sensitive piece of information."

    # Encrypt the data
    encrypted_data = encrypt_data(sensitive_data, key)

    # Save encrypted data to a file (optional)
    with open("encrypted_data.bin", "wb") as f:
        f.write(encrypted_data)

    # Pass seed to the template
    return render(request, 'geocite.html', {'seed': seed})

def decrypt_sensitive_data(seed: str):
    key = generate_key_from_seed(seed)

    # Load the encrypted data from file
    with open("encrypted_data.bin", "rb") as f:
        encrypted_data = f.read()

    # Decrypt the data
    decrypted_data = decrypt_data(encrypted_data, key)
    return decrypted_data

import os
import hashlib
from cryptography.fernet import Fernet

# Define the path to your library folder
LIBRARY_PATH = os.path.join("", "encryted_data")

# Ensure the directory exists
os.makedirs(LIBRARY_PATH, exist_ok=True)

def generate_key_from_seed(seed: str) -> bytes:
    # Convert seed to a 32-byte key using SHA-256 hashing
    sha256 = hashlib.sha256()
    sha256.update(seed.encode())
    return Fernet.generate_key()

def encrypt_data(data: str, key: bytes) -> bytes:
    # Encrypt data
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data: bytes, key: bytes) -> str:
    # Decrypt data
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data)
    return decrypted_data.decode()

def save_encrypted_data(seed: str, data: str):
    # Generate encryption key from the seed
    key = generate_key_from_seed(seed)
    # Encrypt the data
    encrypted_data = encrypt_data(data, key)
    # Define the file path within the library folder
    file_path = os.path.join(LIBRARY_PATH, "encrypted_data.bin")

    # Save encrypted data to a file
    with open(file_path, "wb") as f:
        f.write(encrypted_data)
    print(f"Encrypted data saved to {file_path}")

def load_encrypted_data(seed: str) -> str:
    # Generate encryption key from the seed
    key = generate_key_from_seed(seed)
    # Define the file path within the library folder
    file_path = os.path.join(LIBRARY_PATH, "encrypted_data.bin")

    # Load and decrypt the data from file
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = decrypt_data(encrypted_data, key)
    return decrypted_data