from django.db.models.signals import post_save
from django.dispatch import receiver
from logs.models import LogEntry
from alerts.services import check_alert_rules

@receiver(post_save, sender=LogEntry)
def on_log_saved(sender, instance, created, **kwargs):
    if created:
        check_alert_rules(instance)