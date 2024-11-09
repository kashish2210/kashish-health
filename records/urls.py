from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.auth import views as dashboard
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload/', views.upload_record, name='upload_record'),
    path('view/', views.view_records, name='view_records'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('store_data/', views.store_data, name='store_data'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)