from django.urls import path
from .views import ServerDashboardView, ServerDataAPI, DownloadLogsPDF, ExecuteSQLView

app_name = 'server'

urlpatterns = [
    path('dashboard/', ServerDashboardView.as_view(), name='dashboard'),
    path('api/data/', ServerDataAPI.as_view(), name='data_api'),
    path('download/pdf/', DownloadLogsPDF.as_view(), name='download_pdf'),
    path('execute-sql/', ExecuteSQLView.as_view(), name='execute_sql'),
]
