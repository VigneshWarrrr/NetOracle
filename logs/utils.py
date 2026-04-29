from alerts.services import IDSEngine

def parse_log_file(file, format, user=None):
    """
    Parses a log file (CSV, JSON, TXT) using the IDSEngine.
    Returns the number of logs created and alerts triggered.
    """
    logs_created, alerts_triggered = IDSEngine.process_file(file, format, user=user)
    return logs_created, alerts_triggered
