from alerts.models import Alert, DetectionRule
from alerts.notifications import send_notification
from logs.models import LogEntry, LogSource
from django.utils import timezone
from datetime import timedelta
import re
import csv
import json
import io
import PyPDF2
import os

class IDSEngine:
    """
    The core Intrusion Detection System engine.
    Supports API data, CSV files, and JSON files.
    """

    @staticmethod
    def calculate_anomaly_score(log_entry):
        # Look for spikes in the same event type from the same IP
        since = timezone.now() - timedelta(hours=1)
        recent_count = LogEntry.objects.filter(
            src_ip=log_entry.src_ip,
            event_type=log_entry.event_type,
            timestamp__gte=since
        ).count()
        
        # Simple logic: more than 20 events in an hour is anomalous
        score = min(recent_count / 20.0, 1.0)
        return score

    @classmethod
    def analyze_log(cls, log_entry):
        """
        Analyzes a single LogEntry and triggers alerts if necessary.
        Returns the Alert object if triggered, else None.
        """
        score = cls.calculate_anomaly_score(log_entry)
        
        # Check Anomaly Score
        if score > 0.8:
            alert = Alert.objects.create(
                user=log_entry.user,
                log_entry=log_entry, 
                rule_triggered="AI Anomaly Detection",
                anomaly_score=score
            )
            send_notification(log_entry)
            return alert

        # Check Dynamic Rules
        active_rules = DetectionRule.objects.filter(is_active=True)
        for rule in active_rules:
            triggered = False
            
            if rule.rule_type == 'SEVERITY':
                if log_entry.severity == rule.pattern:
                    triggered = True
            
            elif rule.rule_type == 'REGEX':
                if re.search(rule.pattern, log_entry.message) or re.search(rule.pattern, log_entry.event_type):
                    triggered = True
            
            elif rule.rule_type == 'THRESHOLD':
                since = timezone.now() - timedelta(minutes=rule.time_window)
                count = LogEntry.objects.filter(
                    src_ip=log_entry.src_ip,
                    event_type=rule.pattern,
                    timestamp__gte=since
                ).count()
                if count >= rule.threshold:
                    triggered = True

            if triggered:
                alert = Alert.objects.create(
                    user=log_entry.user,
                    log_entry=log_entry, 
                    rule_triggered=rule.name,
                    anomaly_score=score
                )
                send_notification(log_entry)
                return alert
        
        return None

    @classmethod
    def process_file(cls, file_obj, file_format, user=None):
        """
        Processes an uploaded file and returns a summary of results.
        """
        logs_created = 0
        alerts_triggered = 0
        
        default_source, _ = LogSource.objects.get_or_create(
            name="Manual Upload",
            defaults={'ip_address': '127.0.0.1', 'source_type': 'Manual'}
        )

        try:
            if file_format == 'pdf':
                reader = PyPDF2.PdfReader(file_obj)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Treat extracted text as lines
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    src_ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
                    severity_match = re.search(r'(CRITICAL|HIGH|MEDIUM|LOW)', line, re.IGNORECASE)
                    
                    log = LogEntry.objects.create(
                        user=user,
                        source=default_source,
                        severity=severity_match.group(1).upper() if severity_match else 'MEDIUM',
                        event_type='PDF Log',
                        src_ip=src_ip_match.group(1) if src_ip_match else '0.0.0.0',
                        dest_ip='0.0.0.0',
                        message=line,
                        raw_log=json.dumps({"raw": line})
                    )
                    if cls.analyze_log(log):
                        alerts_triggered += 1
                    logs_created += 1
                
                return logs_created, alerts_triggered

            decoded_file = file_obj.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            
            if file_format == 'csv':
                reader = csv.DictReader(io_string)
                for row in reader:
                    log = LogEntry.objects.create(
                        user=user,
                        source=default_source,
                        severity=row.get('severity', 'MEDIUM').upper(),
                        event_type=row.get('event_type', 'Unknown'),
                        src_ip=row.get('src_ip', '0.0.0.0'),
                        dest_ip=row.get('dest_ip', '0.0.0.0'),
                        message=row.get('message', ''),
                        raw_log=json.dumps(row)
                    )
                    if cls.analyze_log(log):
                        alerts_triggered += 1
                    logs_created += 1
            
            elif file_format == 'json':
                try:
                    data = json.load(io_string)
                    if not isinstance(data, list):
                        data = [data]
                    for item in data:
                        log = LogEntry.objects.create(
                            user=user,
                            source=default_source,
                            severity=item.get('severity', 'MEDIUM').upper(),
                            event_type=item.get('event_type', 'Unknown'),
                            src_ip=item.get('src_ip', '0.0.0.0'),
                            dest_ip=item.get('dest_ip', '0.0.0.0'),
                            message=item.get('message', ''),
                            raw_log=json.dumps(item)
                        )
                        if cls.analyze_log(log):
                            alerts_triggered += 1
                        logs_created += 1
                except json.JSONDecodeError:
                    # If JSON fails, fall back to TXT parsing
                    io_string.seek(0)
                    return cls.process_file(file_obj, 'txt', user=user)
            
            elif file_format == 'txt':
                for line in io_string:
                    line = line.strip()
                    if not line:
                        continue
                    
                    src_ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
                    severity_match = re.search(r'(CRITICAL|HIGH|MEDIUM|LOW)', line, re.IGNORECASE)
                    
                    log = LogEntry.objects.create(
                        user=user,
                        source=default_source,
                        severity=severity_match.group(1).upper() if severity_match else 'MEDIUM',
                        event_type='Text Log',
                        src_ip=src_ip_match.group(1) if src_ip_match else '0.0.0.0',
                        dest_ip='0.0.0.0',
                        message=line,
                        raw_log=json.dumps({"raw": line})
                    )
                    if cls.analyze_log(log):
                        alerts_triggered += 1
                    logs_created += 1
        except Exception as e:
            # General catch to prevent server crash
            print(f"Error processing file: {e}")
            pass
        
        return logs_created, alerts_triggered

def check_alert_rules(log_entry):
    """
    Legacy wrapper for the IDSEngine.
    """
    return IDSEngine.analyze_log(log_entry)