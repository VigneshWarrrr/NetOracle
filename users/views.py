from django.shortcuts import render, redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.contrib import messages
from logs.utils import parse_log_file
from logs.models import LogEntry
from management.models import UserActivity

class UserDashboardView(LoginRequiredMixin, View):
    template_name = 'users/dashboard.html'

    def get(self, request):
        recent_logs = LogEntry.objects.filter(source__name="Manual Upload").order_by('-timestamp')[:5]
        return render(request, self.template_name, {'recent_logs': recent_logs})

    def post(self, request):
        files = request.FILES.getlist('log_file')
        if not files:
            messages.error(request, "No files uploaded.")
            return redirect('users:dashboard')

        total_logs = 0
        total_alerts = 0
        file_names = []

        for file in files:
            file_ext = file.name.split('.')[-1].lower()
            if file_ext in ['csv', 'json', 'txt']:
                logs_count, alerts_count = parse_log_file(file, file_ext)
                total_logs += logs_count
                total_alerts += alerts_count
                file_names.append(file.name)
            else:
                messages.error(request, f"Skipped {file.name}: Unsupported format.")

        if total_logs > 0:
            # Log activity
            UserActivity.objects.create(
                user=request.user, 
                action="Uploaded Logs", 
                details=f"Uploaded {len(file_names)} files: {', '.join(file_names)} ({total_logs} entries, {total_alerts} alerts)"
            )
            
            if total_alerts > 0:
                messages.warning(request, f"Processed {total_logs} entries from {len(file_names)} files. ATTENTION: {total_alerts} security alerts triggered!")
            else:
                messages.success(request, f"Successfully processed {total_logs} entries from {len(file_names)} files. No threats detected.")
        
        return redirect('users:dashboard')
