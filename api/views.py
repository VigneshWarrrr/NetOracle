from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.generic import TemplateView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from .serializers import LogEntrySerializer
from alerts.services import check_alert_rules, IDSEngine
from .models import ExternalDatabase
from logs.models import LogEntry, LogSource
from alerts.models import Alert
import random

from rest_framework.permissions import IsAuthenticated

class LogIngestAPIView(APIView):
    """
    API endpoint for real-time log ingestion.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = LogEntrySerializer(data=request.data)
        if serializer.is_valid():
            log = serializer.save(user=request.user)
            check_alert_rules(log)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class APIDashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'api/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['connectors'] = ExternalDatabase.objects.filter(user=self.request.user)
        # Mock "database table" logs for the first connector
        if context['connectors'].exists():
            connector = context['connectors'].first()
            context['external_logs'] = LogEntry.objects.filter(event_type__icontains='API Sync')[:15]
        return context

class AddConnectorView(LoginRequiredMixin, View):
    def post(self, request):
        db_type = request.POST.get('db_type')
        endpoint = request.POST.get('endpoint')
        api_key = request.POST.get('api_key')
        sync_mode = request.POST.get('sync_mode')

        connector = ExternalDatabase.objects.create(
            user=request.user,
            db_type=db_type,
            endpoint=endpoint,
            api_key=api_key,
            sync_mode=sync_mode
        )

        # MOCK SYNC PROCESS: Generate some logs and alerts
        source, _ = LogSource.objects.get_or_create(
            name=f"External {connector.get_db_type_display()}",
            defaults={'ip_address': '8.8.8.8', 'source_type': 'Cloud'}
        )

        mock_messages = [
            "SQL Injection attempt detected from remote source",
            "Successful database backup via API",
            "Unauthorized access attempt to sensitive table",
            "Schema synchronization completed",
            "High latency detected in remote query"
        ]

        for i in range(5):
            msg = random.choice(mock_messages)
            severity = 'HIGH' if 'attempt' in msg or 'Unauthorized' in msg else 'LOW'
            log = LogEntry.objects.create(
                user=request.user,
                source=source,
                severity=severity,
                event_type='API Sync',
                src_ip='8.8.8.8',
                dest_ip='127.0.0.1',
                message=msg
            )
            # Generate alerts for high severity
            if severity == 'HIGH':
                Alert.objects.create(
                    log_entry=log,
                    user=request.user,
                    rule_triggered=f"Cloud Protection: {msg}",
                    status='Active',
                    severity=severity
                )

        return redirect('api:dashboard')

class DeleteConnectorView(LoginRequiredMixin, View):
    def post(self, request, pk):
        connector = ExternalDatabase.objects.get(pk=pk, user=request.user)
        connector.delete()
        return redirect('api:dashboard')

class UpdateConnectorView(LoginRequiredMixin, View):
    def post(self, request, pk):
        connector = ExternalDatabase.objects.get(pk=pk, user=request.user)
        connector.db_type = request.POST.get('db_type')
        connector.endpoint = request.POST.get('endpoint')
        connector.sync_mode = request.POST.get('sync_mode')
        if request.POST.get('api_key'):
            connector.api_key = request.POST.get('api_key')
        connector.save()
        return redirect('api:dashboard')

class ExecuteRemoteQueryView(LoginRequiredMixin, View):
    def post(self, request):
        connector_id = request.POST.get('connector_id')
        query = request.POST.get('query', '').strip()
        
        try:
            connector = ExternalDatabase.objects.get(pk=connector_id, user=request.user)
            db_type = connector.db_type
            
            # Simulate remote execution
            if db_type in ['mysql', 'postgres', 'oracle']:
                if not query.upper().startswith(('SELECT', 'SHOW', 'DESCRIBE', 'INSERT', 'UPDATE')):
                    return JsonResponse({'error': f'Invalid {db_type.upper()} command. Please use standard SQL.'}, status=400)
                
                # Mock result for SQL
                columns = ['id', 'name', 'status', 'last_updated']
                rows = [
                    [1, 'Production_Data', 'Active', '2026-04-29'],
                    [2, 'Backup_Index', 'Idle', '2026-04-28'],
                    [3, 'Temp_Cache', 'Expiring', '2026-04-29']
                ]
                return JsonResponse({
                    'db_type': db_type,
                    'columns': columns,
                    'rows': rows,
                    'message': f'Simulated {db_type.upper()} query executed at {connector.endpoint}'
                })
            
            elif db_type == 'mongo':
                if not (query.startswith('{') or query.startswith('db.')):
                    return JsonResponse({'error': 'Invalid MongoDB command. Please use JSON or db.collection format.'}, status=400)
                
                # Mock result for Mongo
                columns = ['_id', 'document_name', 'metadata', 'version']
                rows = [
                    ['64a7...', 'config_v1', '{"env": "prod"}', 1.0],
                    ['64b8...', 'user_profiles', '{"count": 1024}', 2.1]
                ]
                return JsonResponse({
                    'db_type': db_type,
                    'columns': columns,
                    'rows': rows,
                    'message': f'Simulated MongoDB aggregate executed at {connector.endpoint}'
                })

        except ExternalDatabase.DoesNotExist:
            return JsonResponse({'error': 'Connector not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)