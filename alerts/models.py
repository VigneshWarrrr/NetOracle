from django.contrib.auth.models import User
from django.db import models

from logs.models import LogEntry


class DetectionRule(models.Model):
    name = models.CharField(max_length=150)
    pattern = models.CharField(max_length=255)
    enabled = models.BooleanField(default=True)


class Alert(models.Model):
    log_entry = models.ForeignKey(LogEntry, on_delete=models.CASCADE, null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    rule_triggered = models.CharField(max_length=255)
    status = models.CharField(max_length=20, default="Active")
    severity = models.CharField(max_length=10, default="MEDIUM")
    notified = models.BooleanField(default=False)
    anomaly_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
