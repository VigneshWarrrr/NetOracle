from django.db import models
from django.contrib.auth.models import User

class LogSource(models.Model):
    name = models.CharField(max_length=100)        # e.g., "Apache Server", "Firewall-01"
    ip_address = models.GenericIPAddressField()
    source_type = models.CharField(max_length=50) # e.g., "IDS", "Firewall", "Server"

class LogEntry(models.Model):
    SEVERITY_CHOICES = [('LOW','Low'), ('MEDIUM','Medium'),
                        ('HIGH','High'), ('CRITICAL','Critical')]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='user_log_entries')
    source = models.ForeignKey(LogSource, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    event_type = models.CharField(max_length=100)  # e.g., "Port Scan", "SQL Injection"
    src_ip = models.GenericIPAddressField()
    dest_ip = models.GenericIPAddressField()
    message = models.TextField()
    raw_log = models.TextField()                   # original unparsed log
    is_flagged = models.BooleanField(default=False)
