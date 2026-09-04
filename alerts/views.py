from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView

from .models import Alert


class AlertDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "alerts/dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["alerts"] = Alert.objects.order_by("-created_at")[:50]
        return context
