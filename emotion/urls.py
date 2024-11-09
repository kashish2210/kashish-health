from django.urls import path
from . import views

urlpatterns = [
    path('video_feed/', views.webcam_feed, name='video_feed'),
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),
    path('graph/', views.generate_graph, name='generate_graph'),
]
