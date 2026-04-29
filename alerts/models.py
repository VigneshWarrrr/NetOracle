from django.db import models
from logs.models import LogEntry

class DetectionRule(models.Model):
    RULE_TYPES = [
        ('REGEX', 'Regular Expression'),
        ('THRESHOLD', 'Frequency Threshold'),
        ('SEVERITY', 'Severity Based'),
    ]
    
    name = models.CharField(max_length=100)
    rule_type = models.CharField(max_length=20, choices=RULE_TYPES)
    pattern = models.CharField(max_length=255, help_text="Regex pattern or event type name")
    threshold = models.IntegerField(default=1, help_text="Number of occurrences required (for THRESHOLD type)")
    time_window = models.IntegerField(default=5, help_text="Time window in minutes (for THRESHOLD type)")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.rule_type})"

from django.contrib.auth.models import User

class Alert(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='user_alerts')
    log_entry = models.ForeignKey(LogEntry, on_delete=models.CASCADE)
    rule_triggered = models.CharField(max_length=200)
    anomaly_score = models.FloatField(default=0.0)
    notified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='active')

    def __str__(self):
        return f"Alert: {self.rule_triggered} at {self.created_at}"
