from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),  # Route for uploading MRI images
    path('result/<int:pk>/', views.result, name='result'),  # Route for displaying results
]