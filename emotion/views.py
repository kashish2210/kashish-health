
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
import google.generativeai as genai
import json
import random
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

genai.configure(api_key="AIzaSyBxTtDK9FF2bVo5PPmJsHbM_JAViE7X-d0")
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")


import json
import random
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# --- Utility for extracting Gemini response text ---
def extract_text_from_response(response):
    """Safely extract text from Gemini response"""
    if hasattr(response, "text") and response.text:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except Exception:
        return ""


# --- Psychological Question Generator ---
def generate_psychological_questions():
    """Generate psychological questions using Gemini AI"""
    try:
        question_themes = [
            "anxiety and worry patterns",
            "emotional regulation and mood",
            "social relationships and support",
            "work-life balance and stress",
            "self-esteem and confidence",
            "sleep and physical well-being",
            "motivation and life purpose",
            "coping mechanisms and resilience"
        ]

        selected_themes = random.sample(question_themes, 5)

        prompt = f"""
        Generate 5 unique psychological well-being assessment questions in JSON format. 
        Focus on these themes: {', '.join(selected_themes)}.

        Each question should have:
        1. A clear, empathetic, professional question text
        2. 4 answer options with scores (4=best mental health, 1=concerning mental health)
        3. Questions should feel like they come from a caring therapist
        4. Avoid repetitive phrasing - make each question unique
        5. Include varied question styles (frequency, intensity, coping, relationships)

        Return ONLY valid JSON in this exact format:
        {{
            "questions": [
                {{
                    "text": "question text here",
                    "category": "theme category",
                    "options": [
                        {{"text": "option 1", "score": 4}},
                        {{"text": "option 2", "score": 3}},
                        {{"text": "option 3", "score": 2}},
                        {{"text": "option 4", "score": 1}}
                    ]
                }}
            ]
        }}
        """

        response = gemini_model.generate_content(prompt)
        response_text = extract_text_from_response(response)

        if not response_text.strip():
            raise ValueError("Empty response from Gemini")

        try:
            questions_data = json.loads(response_text)
            if "questions" not in questions_data:
                raise ValueError("Response missing 'questions' key")
            return questions_data
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON from Gemini")

    except Exception as e:
        print(f"Error generating questions: {e}")

        # --- Fallback questions (safe minimum set) ---
        fallback_questions = [
            {
                "text": "How often do you feel overwhelmed by daily responsibilities?",
                "category": "stress",
                "options": [
                    {"text": "Rarely or never", "score": 4},
                    {"text": "Sometimes", "score": 3},
                    {"text": "Often", "score": 2},
                    {"text": "Almost always", "score": 1},
                ],
            },
            {
                "text": "How connected do you feel to your support system (friends/family)?",
                "category": "relationships",
                "options": [
                    {"text": "Very connected", "score": 4},
                    {"text": "Somewhat connected", "score": 3},
                    {"text": "Not very connected", "score": 2},
                    {"text": "Not connected at all", "score": 1},
                ],
            },
            {
                "text": "How well do you sleep at night?",
                "category": "sleep",
                "options": [
                    {"text": "Very well, restful sleep", "score": 4},
                    {"text": "Mostly okay, some issues", "score": 3},
                    {"text": "Poor, frequent issues", "score": 2},
                    {"text": "Very poor, hardly restful", "score": 1},
                ],
            },
            {
                "text": "How confident do you feel in handling challenges?",
                "category": "confidence",
                "options": [
                    {"text": "Very confident", "score": 4},
                    {"text": "Fairly confident", "score": 3},
                    {"text": "Sometimes confident", "score": 2},
                    {"text": "Rarely confident", "score": 1},
                ],
            },
            {
                "text": "How motivated do you feel to pursue your goals?",
                "category": "motivation",
                "options": [
                    {"text": "Highly motivated", "score": 4},
                    {"text": "Somewhat motivated", "score": 3},
                    {"text": "Occasionally motivated", "score": 2},
                    {"text": "Not motivated", "score": 1},
                ],
            },
        ]

        return {
            "questions": random.sample(
                fallback_questions, min(5, len(fallback_questions))
            )
        }


# --- AI Therapist Analysis ---
def generate_ai_therapist_analysis(answers_data):
    """Generate personalized analysis using Gemini AI"""
    try:
        total_score = answers_data.get("total_score", 0)
        percentage = answers_data.get("percentage", 0)
        categories = answers_data.get("categories", {})

        prompt = f"""
        You are a compassionate AI therapist analyzing a patient's psychological well-being assessment.

        Assessment Results:
        - Total Score: {total_score}/20 ({percentage:.1f}%)
        - Category breakdown: {categories}

        Provide a personalized, empathetic response that includes:
        1. A warm, professional greeting
        2. Acknowledgment of their current state
        3. 2-3 specific, actionable recommendations based on their responses
        4. Encouraging words about their potential for growth
        5. A gentle reminder about professional support if needed

        Keep response between 100-150 words. Use a caring, therapeutic tone.
        Focus on strengths while addressing areas for improvement.

        Return as plain text, not JSON.
        """

        response = gemini_model.generate_content(prompt)
        response_text = extract_text_from_response(response)
        return response_text.strip() if response_text else "I'm here to support you. Please remember you're not alone, and professional help is always available if needed."

    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        # fallback simplified analysis
        percentage = answers_data.get("percentage", 0)
        if percentage >= 85:
            return "You're showing remarkable resilience and emotional well-being! ..."
        elif percentage >= 70:
            return "You're demonstrating good emotional balance with some areas for gentle attention. ..."
        elif percentage >= 50:
            return "Thank you for your honest responses. You're showing awareness of your emotional state..."
        else:
            return "I appreciate your openness in sharing your experiences. Your responses suggest you may be going through a challenging time..."


# --- Django Views ---
@csrf_exempt
def get_new_questions(request):
    if request.method == "GET":
        questions_data = generate_psychological_questions()
        return JsonResponse(questions_data)
    return JsonResponse({"error": "Invalid request method"})


@csrf_exempt
def analyze_responses(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            answers = data.get("answers", [])

            if len(answers) != 5:
                return JsonResponse({"error": "Please answer all questions"})

            total_score = sum(answer["score"] for answer in answers)
            max_score = 20
            percentage = (total_score / max_score) * 100

            # group by category
            categories = {}
            for answer in answers:
                category = answer.get("category", "general")
                categories.setdefault(category, []).append(answer["score"])

            category_averages = {
                cat: sum(scores) / len(scores) for cat, scores in categories.items()
            }

            analysis_data = {
                "total_score": total_score,
                "percentage": percentage,
                "categories": category_averages,
            }

            ai_analysis = generate_ai_therapist_analysis(analysis_data)

            if percentage >= 85:
                status, status_text = "excellent", "Excellent"
            elif percentage >= 70:
                status, status_text = "good", "Good"
            elif percentage >= 50:
                status, status_text = "fair", "Fair"
            else:
                status, status_text = "poor", "Needs Attention"

            return JsonResponse(
                {
                    "total_score": total_score,
                    "percentage": percentage,
                    "status": status,
                    "status_text": status_text,
                    "ai_analysis": ai_analysis,
                    "category_breakdown": category_averages,
                }
            )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"})
        except Exception as e:
            return JsonResponse({"error": f"Analysis error: {str(e)}"})

    return JsonResponse({"error": "Invalid request method"})


# --- Stress Insights ---
def get_stress_insights(score_percentage):
    if score_percentage >= 85:
        return {
            "level": "Low Stress",
            "color": "#2ed573",
            "recommendations": [
                "Maintain current healthy habits",
                "Continue stress management practices",
                "Share your strategies with others",
            ],
        }
    elif score_percentage >= 70:
        return {
            "level": "Mild Stress",
            "color": "#5352ed",
            "recommendations": [
                "Practice deep breathing exercises",
                "Ensure adequate sleep (7-9 hours)",
                "Engage in regular physical activity",
            ],
        }
    elif score_percentage >= 50:
        return {
            "level": "Moderate Stress",
            "color": "#ffa502",
            "recommendations": [
                "Try mindfulness or meditation",
                "Limit caffeine and alcohol",
                "Connect with supportive friends/family",
            ],
        }
    else:
        return {
            "level": "High Stress",
            "color": "#ff4757",
            "recommendations": [
                "Consider professional counseling",
                "Practice stress-reduction techniques daily",
                "Prioritize self-care and rest",
            ],
        }

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
