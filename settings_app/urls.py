from django.urls import path
from .views import SettingsDashboardView

app_name = 'settings_app'

urlpatterns = [
    path('', SettingsDashboardView.as_view(), name='dashboard'),
]
