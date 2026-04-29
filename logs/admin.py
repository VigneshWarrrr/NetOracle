from django.contrib import admin
from logs.models import LogEntry, LogSource

@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'event_type', 'severity', 'src_ip')
    list_filter = ('severity', 'event_type')
    search_fields = ('src_ip', 'message')

@admin.register(LogSource)
class LogSourceAdmin(admin.ModelAdmin):
    list_display = ('name', 'ip_address', 'source_type')
