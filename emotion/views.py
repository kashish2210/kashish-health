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
import plotly.graph_objects as go  # Ensure this import is included

DATA_FILE_PATH = 'emotion_data.json'

# Load the trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Define emotion labels
emotion_labels = ['sad', 'happy', 'angry']

# Initialize emotion counts if the file does not exist
if not os.path.exists(DATA_FILE_PATH):
    with open(DATA_FILE_PATH, 'w') as file:
        json.dump({'happy': 0, 'sad': 0, 'angry': 0}, file)

# Video capture generator function
def video_feed():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract face from the frame
                face = frame[y:y + h, x:x + w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)

                # Predict emotion
                predictions = model.predict(face)
                emotion = emotion_labels[np.argmax(predictions)]

                # Update emotion counts in the JSON file
                try:
                    with open(DATA_FILE_PATH, 'r') as file:
                        emotion_counts = json.load(file)
                except (json.JSONDecodeError, IOError) as e:
                    emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}

                emotion_counts[emotion] += 1

                with open(DATA_FILE_PATH, 'w') as file:
                    json.dump(emotion_counts, file)

                # Display emotion
                putTextRect(frame, emotion, (x, y - 10), colorR=(0, 255, 0))

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert to bytes and yield for streaming
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

# Django view to get emotion data
def get_emotion_data(request):
    try:
        with open(DATA_FILE_PATH, 'r') as file:
            emotion_counts = json.load(file)
    except (json.JSONDecodeError, IOError):
        emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}
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
        data = {'happy': 0, 'sad': 0, 'angry': 0}

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
