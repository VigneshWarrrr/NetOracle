from django.urls import path

from . import views

app_name = "alerts"

urlpatterns = [
    path("", views.AlertDashboardView.as_view(), name="dashboard"),
]
