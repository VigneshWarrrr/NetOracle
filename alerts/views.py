from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView
from .models import Alert

class AlertDashboardView(LoginRequiredMixin, ListView):
    model = Alert
    template_name = 'alerts/dashboard.html'
    context_object_name = 'alerts'
    paginate_by = 50

    def get_queryset(self):
        qs = Alert.objects.select_related('log_entry', 'log_entry__source', 'user').order_by('-created_at')
        if not self.request.user.is_staff:
            qs = qs.filter(user=self.request.user)
        return qs
