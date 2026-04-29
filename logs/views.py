from django.shortcuts import render, redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView
from django.contrib import messages
from .models import LogEntry
from .utils import parse_log_file

class LogDashboardView(LoginRequiredMixin, ListView):
    model = LogEntry
    template_name = 'logs/dashboard.html'
    context_object_name = 'logs'
    paginate_by = 50

    def get_queryset(self):
        qs = LogEntry.objects.select_related('source', 'user').order_by('-timestamp')
        
        # Filter by user if not staff
        if not self.request.user.is_staff:
            qs = qs.filter(user=self.request.user)
            
        severity = self.request.GET.get('severity')
        if severity:
            qs = qs.filter(severity=severity)
        src_ip = self.request.GET.get('src_ip')
        if src_ip:
            qs = qs.filter(src_ip=src_ip)
        return qs

    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist('log_file')
        if not files:
            messages.error(request, "No files uploaded.")
            return redirect('logs:dashboard')

        total_logs = 0
        total_alerts = 0
        processed_files = 0

        for file in files:
            file_ext = file.name.split('.')[-1].lower()
            if file_ext in ['csv', 'json', 'txt', 'pdf']:
                logs_count, alerts_count = parse_log_file(file, file_ext, user=request.user)
                total_logs += logs_count
                total_alerts += alerts_count
                processed_files += 1
            else:
                messages.error(request, f"Skipped {file.name}: Unsupported format.")

        if total_logs > 0:
            if total_alerts > 0:
                messages.warning(request, f"Processed {total_logs} entries from {processed_files} files. {total_alerts} security alerts triggered!")
            else:
                messages.success(request, f"Successfully processed {total_logs} entries from {processed_files} files.")
            
        return redirect('logs:dashboard')