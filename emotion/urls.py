from django.urls import path
from . import views
from users import views as user_views  # Import the chatbot view from the users app

urlpatterns = [
    path('video_feed/', views.webcam_feed, name='video_feed'),  # Updated function name
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),
    path('graph/', views.generate_graph, name='generate_graph'),
    path('detect/', views.detect, name='detect'),
    path('alert/', views.alert, name='alert'),
    path('alert/chatbot//', user_views.chatbot_view, name='chatbot'),
    path('alert/multilingual/', views.multilingual, name='multilingual'),
    path('geocite/', views.geocite_interface, name='geocite_interface'),
]
