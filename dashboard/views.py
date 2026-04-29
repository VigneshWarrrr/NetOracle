from django.shortcuts import render, redirect

from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from logs.models import LogEntry, LogSource
from alerts.models import Alert

class AdminDashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/index.html'

    def dispatch(self, request, *args, **kwargs):
        if not (request.user.is_authenticated):
            return redirect('users:login')
        if not (request.user.is_staff or request.user.is_superuser):
            return redirect('users:dashboard')
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_logs'] = LogEntry.objects.all().order_by('-timestamp')[:10]
        context['all_alerts'] = Alert.objects.all().order_by('-created_at')[:10]
        context['log_sources'] = LogSource.objects.all()
        context['total_logs'] = LogEntry.objects.count()
        context['active_alerts'] = Alert.objects.filter(status='active').count() if hasattr(Alert, 'status') else Alert.objects.count()
        return context
