from django.contrib import admin
from alerts.models import Alert

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'rule_triggered', 'status', 'notified')
    list_filter = ('status', 'rule_triggered')
