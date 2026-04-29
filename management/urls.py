from django.urls import path
from . import views

app_name = 'management'

urlpatterns = [
    path('dashboard/', views.UserManagementDashboardView.as_view(), name='dashboard'),
    path('create/', views.UserCreateView.as_view(), name='user_create'),
    path('delete/<int:user_id>/', views.UserDeleteView.as_view(), name='user_delete'),
]
