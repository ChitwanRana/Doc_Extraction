from django.urls import path
from .views import extract_view

urlpatterns = [
    path('', extract_view, name='extract_view'),
]
