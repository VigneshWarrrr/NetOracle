from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('users.urls')),
    path('dashboard/', include('dashboard.urls')),
    path('logs/', include('logs.urls')),
    path('alerts/', include('alerts.urls')),
    path('api/', include('api.urls')),
    path('management/', include('management.urls')),
    path('server/', include('server.urls')),
    path('settings/', include('settings_app.urls')),
    path('login/login/', RedirectView.as_view(url='/users/login/')), # Safety redirect for the 404 path
    path('', RedirectView.as_view(url='/users/login/')),
]
