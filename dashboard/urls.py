from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.AdminDashboardView.as_view(), name='index'),
    path('forecast/', views.NetworkRiskForecastView.as_view(), name='forecast'),
]
