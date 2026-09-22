from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('dashboard/', views.APIDashboardView.as_view(), name='dashboard'),
    path('ingest/', views.LogIngestAPIView.as_view(), name='ingest'),
    path('add-connector/', views.AddConnectorView.as_view(), name='add_connector'),
    path('delete-connector/<int:pk>/', views.DeleteConnectorView.as_view(), name='delete_connector'),
    path('update-connector/<int:pk>/', views.UpdateConnectorView.as_view(), name='update_connector'),
    path('execute-remote-query/', views.ExecuteRemoteQueryView.as_view(), name='execute_remote_query'),
    path('authoritative-predict/', views.AuthoritativeForecastAPIView.as_view(), name='authoritative_predict'),
]
