from django.db import models
from django.contrib.auth.models import User

class ExternalDatabase(models.Model):
    DB_TYPES = [
        ('mysql', 'MySQL / MariaDB'),
        ('postgres', 'PostgreSQL'),
        ('mongo', 'MongoDB'),
        ('oracle', 'Oracle'),
    ]
    
    SYNC_MODES = [
        ('push', 'Continuous Push'),
        ('poll', 'Interval Polling (1m)'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    db_type = models.CharField(max_length=20, choices=DB_TYPES)
    endpoint = models.URLField()
    api_key = models.CharField(max_length=255)
    sync_mode = models.CharField(max_length=10, choices=SYNC_MODES)
    status = models.CharField(max_length=20, default='Connected')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_db_type_display()} - {self.endpoint}"
