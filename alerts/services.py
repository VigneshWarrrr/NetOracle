from pathlib import Path

from django.utils import timezone

from alerts.models import Alert
from logs.models import LogEntry, LogSource


class IDSEngine:
    """Small IDS adapter used by log ingestion and server monitoring."""

    HIGH_RISK_TERMS = ("attack", "injection", "unauthorized", "brute", "malware", "scan")

    @classmethod
    def process_file(cls, file, format="txt", user=None):
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        lines = [line.strip() for line in str(content).splitlines() if line.strip()]
        source, _ = LogSource.objects.get_or_create(
            name="IDS upload",
            defaults={"ip_address": "127.0.0.1", "source_type": "IDS"},
        )
        created = 0
        alerts = 0
        for line in lines:
            severity = "HIGH" if any(term in line.lower() for term in cls.HIGH_RISK_TERMS) else "LOW"
            entry = LogEntry.objects.create(
                user=user,
                source=source,
                severity=severity,
                event_type="IDS event",
                src_ip="127.0.0.1",
                dest_ip="127.0.0.1",
                message=line[:500],
                raw_log=line,
            )
            created += 1
            if check_alert_rules(entry):
                alerts += 1
        return created, alerts


def check_alert_rules(log):
    if str(log.severity).upper() not in {"HIGH", "CRITICAL"}:
        return None
    alert, _ = Alert.objects.get_or_create(
        log_entry=log,
        defaults={
            "user": log.user,
            "rule_triggered": f"High severity IDS event: {log.event_type}",
            "status": "Active",
            "severity": log.severity,
            "anomaly_score": 1.0,
        },
    )
    return alert
