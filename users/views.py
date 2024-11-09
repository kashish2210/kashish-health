from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.views.generic import TemplateView
from django.contrib.auth.views import LoginView as AuthLoginView
from django.urls import reverse_lazy
from django import forms
import json
from django.template import loader
import os
import plotly.graph_objects as go
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai

apiKey = "AIzaSyA2i4KZU4YzGty_GN0-obC07e_ufWPlxdg"

DATA_FILE_PATH = 'emotion_data.json'
# Custom User Model
User = get_user_model()

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2')

class CustomAuthenticationForm(AuthenticationForm):
    class Meta:
        model = User
        fields = ('username', 'password')

class CustomLoginView(AuthLoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True

def register_patient(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_doctor = False
            user.save()
            auth_login(request, user)  # Log in after registration
            return redirect('dashboard')  # Redirect to the dashboard
    else:
        form = CustomUserCreationForm()
    return render(request, 'register_patient.html', {'form': form})

def register_doctor(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_doctor = True
            user.save()
            auth_login(request, user)  # Log in after registration
            return redirect('dashboard')  # Redirect to the dashboard
    else:
        form = CustomUserCreationForm()
    return render(request, 'register_doctor.html', {'form': form})

@login_required(login_url='login')
def dashboard(request):
    # Check if reset button was clicked
    if 'reset' in request.GET:
        # Reset the data (for example, reset emotion counts to 0)
        data = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'demotivated': 0}
        
        # Write the reset data back to the file (assuming you're using a JSON file)
        with open(DATA_FILE_PATH, 'w') as file:
            json.dump(data, file)

        # Redirect to the dashboard to refresh the graphs
        return HttpResponseRedirect(reverse('dashboard'))

    # Read data from the JSON file
    try:
        with open(DATA_FILE_PATH, 'r') as file:
            data = json.load(file)
    except (json.JSONDecodeError, IOError):
        data = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'demotivated': 0}

    emotions = list(data.keys())
    counts = list(data.values())

    # Get the angry count (assuming it exists in the data)
    angry_count = data.get('angry', 0)

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

    # Check if the user is a patient and logged in
    if request.user.is_authenticated and not request.user.is_doctor:
        # If angry count is above 100, pass it to the template
        alert_triggered = angry_count > 100
    else:
        alert_triggered = False

    # Render the HTML template with the graphs and alert trigger
    context = {
        'graph_html_2d': graph_html_2d,
        'graph_html_3d': graph_html_3d,
        'alert_triggered': alert_triggered,  # Pass the alert status to the template
        'angry_count': angry_count
    }

    return render(request, 'dashboard.html', context)
def register_selection(request):
    return render(request, 'register_selection.html')


@csrf_exempt
def summarize(request):
    raw_data = request.body
    article = raw_data.decode('utf-8')

    numOfWords = 150
    prompt = (
        f"Summarize this article in {numOfWords} words. I am using the result of this prompt for an webapp. So make sure that it doesn't contain any formatting. It should be plain text. Here is the article: "
        + article
    )

    genai.configure(api_key=apiKey)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])

    response = chat_session.send_message(prompt)

    jsonResult = {"result" : response.text}
    
    return JsonResponse(jsonResult)


from django.core.files.storage import FileSystemStorage

from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def upload(request):
    return render(request, 'fileupload.html')

def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)
        file_path = fs.path(filename)

        with open(file_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text+= page.extract_text()

        numOfWords = 150
        prompt = (
            f"Summarize this pdf files text in {numOfWords} words. I am using the result of this prompt for an webapp. So make sure that it doesn't contain any formatting. It should be plain text. Here is the article: "
            + text
        )

        genai.configure(api_key=apiKey)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(history=[])

        response = chat_session.send_message(prompt)
        return HttpResponse(response.text)
    return render(request, 'fileupload.html')


from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai
import os
import json
from django.http import JsonResponse


chat_history = []
genai.configure(api_key="AIzaSyA2i4KZU4YzGty_GN0-obC07e_ufWPlxdg")

model = genai.GenerativeModel("gemini-1.5-flash")

@csrf_exempt
def chatbot_view(request):
    global chat_history  # Access the global chat history

    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("user_message")

        if user_message:

            prompt = f""""Here's a question from a user who may be struggling with mental health. '{user_message}'. 
              The user might be expressing distress, sadness, confusion, or other sensitive emotions. 
              Your response should be empathetic, supportive, and non-judgmental, treating the user with care and understanding. 
              Offer gentle encouragement, listen actively, and provide comforting and professional advice where appropriate. 
              You are a virtual therapist, not a bot. Your goal is to provide mental health support with compassion and kindness.

              Here is the history of the conversation, including previous responses from both the user and the therapist:
              {str(chat_history)}. Make sure that the tone remains respectful, kind, and professional. 
              Your response should address the user's concerns and provide support in a therapeutic manner. 
              The goal is not to solve everything but to provide a safe space and guide the user to feel heard and understood."""

            # Generate the response using the model
            response = model.generate_content(prompt)
            # Add the user message to chat history
            chat_history.append({"user": "user", "text": user_message})

            # Extract the response from the model output
            bot_response = response.text

            # Add the bot response to chat history
            chat_history.append({"user": "bot", "text": bot_response})

        # Return the generated response in JSON format
        return JsonResponse({"response": bot_response})

    # Pass the chat history to the template
    return render(request, "chatbot.html", {"chat_history": chat_history})

