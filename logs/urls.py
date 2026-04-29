from django.urls import path
from . import views

app_name = 'logs'

urlpatterns = [
    path('dashboard/', views.LogDashboardView.as_view(), name='dashboard'),
]
