from django.urls import path
from . import views
from django.contrib.auth import views as auth_views# Ensure this import



urlpatterns = [
    # Login view using Django's built-in LoginView
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    # Registration and other views
    path('register/', views.register_selection, name='register_selection'),
    path('register/patient/', views.register_patient, name='register_patient'),  # Patient registration
    path('register/doctor/', views.register_doctor, name='register_doctor'),  # Doctor registration
    path('dashboard/', views.dashboard, name='home'),  # Dashboard view
    path('logout/', auth_views.LoginView.as_view(template_name='login.html'), name='logout'),
    path("summarize", views.summarize, name = "summarize"),
    path('uploadfile', views.upload, name='upload'),
    path('upload-file', views.upload_file, name='upload_file'),  # Logout view
]
