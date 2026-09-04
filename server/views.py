from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import TemplateView
from django.views import View
from django.conf import settings
import os
import re

from alerts.services import IDSEngine

from logs.models import LogEntry, LogSource
from alerts.models import Alert, DetectionRule
from django.contrib.auth.models import User

class AdminOnlyMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_staff or self.request.user.is_superuser

class ServerDashboardView(LoginRequiredMixin, AdminOnlyMixin, TemplateView):
    template_name = 'server/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        log_file = os.path.join(settings.BASE_DIR, 'server.log')
        
        server_logs = []
        if os.path.exists(log_file):
            # Run IDS Engine on the server log file
            engine = IDSEngine()
            with open(log_file, 'rb') as f:
                engine.process_file(f, 'txt', user=self.request.user)
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                server_logs = lines[-100:]
                server_logs.reverse()
        
        context['server_logs'] = server_logs
        context['db_size'] = os.path.getsize(os.path.join(settings.BASE_DIR, 'db.sqlite3')) / (1024 * 1024)
        
        # Actual Content Stats
        context['stats'] = {
            'total_logs': LogEntry.objects.count(),
            'total_alerts': Alert.objects.count(),
            'total_users': User.objects.count(),
            'total_sources': LogSource.objects.count(),
            'detection_rules': DetectionRule.objects.count()
        }
        
        context['raw_logs'] = list(LogEntry.objects.all().order_by('-timestamp')[:20].values(
            'id', 'user__username', 'timestamp', 'src_ip', 'event_type', 'severity', 'message'
        ))
        context['raw_alerts'] = list(Alert.objects.all().order_by('-created_at')[:20].values(
            'id', 'user__username', 'created_at', 'rule_triggered', 'status', 'notified', 'anomaly_score'
        ))
        
        return context

from django.http import JsonResponse, HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

from django.utils import timezone

class DownloadLogsPDF(AdminOnlyMixin, View):
    def get(self, request, *args, **kwargs):
        # SYNC LATEST LOGS FROM SERVER.LOG BEFORE GENERATING PDF
        log_file = 'server.log'
        if os.path.exists(log_file):
            engine = IDSEngine()
            with open(log_file, 'rb') as f:
                engine.process_file(f, 'txt', user=request.user)

        # Create a file-like buffer to receive PDF data.
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        title = Paragraph("<b>NetOracle: Latest 100 Logs Report</b>", styles['Title'])
        elements.append(title)
        
        # Display generation time in local timezone
        now_local = timezone.localtime(timezone.now())
        elements.append(Paragraph(f"<font color='#666'>Report Generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (Local Time)</font>", styles['Normal']))
        elements.append(Paragraph("<br/>", styles['Normal']))

        # Data - Get latest 100
        logs = LogEntry.objects.all().order_by('-timestamp')[:100]
        data = [['ID', 'User', 'Timestamp', 'IP Address', 'Event', 'Severity']]
        
        for log in logs:
            # Format each log timestamp to local time
            log_local_time = timezone.localtime(log.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            data.append([
                str(log.id),
                log.user.username if log.user else 'System',
                log_local_time,
                log.src_ip,
                log.event_type[:20],
                str(log.severity)
            ])

        # Table
        table = Table(data, colWidths=[40, 80, 110, 100, 110, 60])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38bdf8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        elements.append(table)
        doc.build(elements)

        # File response with dynamic timestamped filename
        buffer.seek(0)
        filename = f"loggedin_logs_{now_local.strftime('%Y%m%d_%H%M')}.pdf"
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
        return response

class ServerDataAPI(AdminOnlyMixin, View):
    def get(self, request, *args, **kwargs):
        # ACTIVE SYNC: Process latest logs from file before returning data
        log_file = 'server.log'
        if os.path.exists(log_file):
            engine = IDSEngine()
            with open(log_file, 'rb') as f:
                engine.process_file(f, 'txt', user=request.user)

        server_logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                server_logs = f.readlines()[-20:]
        
        db_path = 'db.sqlite3'
        db_size = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0

        stats = {
            'total_logs': LogEntry.objects.count(),
            'total_alerts': Alert.objects.count(),
            'total_users': User.objects.count(),
            'total_sources': LogSource.objects.count(),
            'detection_rules': DetectionRule.objects.count()
        }

        raw_logs = list(LogEntry.objects.all().order_by('-timestamp')[:20].values(
            'id', 'user__username', 'timestamp', 'src_ip', 'event_type', 'severity', 'message'
        ))
        
        raw_alerts = list(Alert.objects.all().order_by('-created_at')[:20].values(
            'id', 'user__username', 'created_at', 'rule_triggered', 'status', 'notified', 'anomaly_score'
        ))

        return JsonResponse({
            'server_logs': server_logs,
            'db_size': round(db_size, 2),
            'stats': stats,
            'raw_logs': raw_logs,
            'raw_alerts': raw_alerts
        })

from django.db import connection

class ExecuteSQLView(AdminOnlyMixin, View):
    def post(self, request):
        sql = request.POST.get('sql', '').strip()
        if not sql:
            return JsonResponse({'error': 'Empty query'}, status=400)
        
        # Security check: Block modification of logs_logentry
        forbidden_patterns = [
            r'UPDATE\s+logs_logentry',
            r'DELETE\s+FROM\s+logs_logentry',
            r'DROP\s+TABLE\s+logs_logentry',
            r'TRUNCATE\s+logs_logentry',
            r'ALTER\s+TABLE\s+logs_logentry'
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return JsonResponse({'error': 'Permission Denied: Log table is read-only.'}, status=403)

        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    return JsonResponse({
                        'columns': columns,
                        'rows': rows,
                        'message': f'Success: {len(rows)} rows returned.'
                    })
                return JsonResponse({'message': 'Query executed successfully (No results returned).'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
