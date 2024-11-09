from django.contrib import admin
from django.urls import path, include
from users import views as user_views
from records import views as record_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', user_views.dashboard, name='home'),  # Set dashboard as the home page
    path('', include('users.urls')),  # Include users app urls
    path('', include('records.urls')),
    path('', include('emotion.urls')),
]
