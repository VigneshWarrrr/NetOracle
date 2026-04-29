# Loggedin: Advanced Log Management & Security Intelligence Platform

**Loggedin** is a state-of-the-art administrative dashboard and security monitoring platform designed to provide real-time visibility into system logs, security threats, and external database integrations. Built with a focus on high-performance monitoring and premium user experience, Loggedin empowers administrators to manage their entire infrastructure from a unified, secure interface.

## 🚀 Key Features

### 📊 Comprehensive Dashboard
- **Real-Time Analytics**: Monitor system health, user activity, and security status at a glance.
- **Dynamic Metrics**: Live-updating cards for total logs, active alerts, and database performance.

### 🛡️ Security & IDS Engine
- **Intrusion Detection System (IDS)**: Automatic scanning of system logs for malicious patterns.
- **PDF Ingestion**: Support for analyzing threats within document-based log files using PyPDF2.
- **Anomaly Scoring**: AI-driven risk assessment for detected security events.

### 🖥️ Server Monitoring
- **Live Console**: Real-time stream of the `server.log` file directly in your browser.
- **Interactive SQL Terminal**: Execute raw SQL queries safely against the `db.sqlite3` database.
- **Log Table Browser**: Interactive database table view with full audit history.
- **Export Reports**: Generate professional PDF audit reports with localized timestamps.

### 🔌 API Engine & External Connectors
- **Multi-Database Support**: Connect to external MySQL, PostgreSQL, Oracle, and MongoDB sources.
- **Remote Terminal**: Interact with external databases using context-aware query environments (SQL or NoSQL).
- **Automated Sync**: Schedule interval polling or continuous push for remote log ingestion.

### 👥 User & Management
- **Admin Authentication**: Secure login system with role-based access control.
- **User Management**: Unified interface to manage platform administrators and permissions.

---

## 🛠️ Tech Stack
- **Backend**: Django (Python)
- **Frontend**: HTML5, Vanilla JS, Premium Modern CSS
- **Database**: SQLite (Core), support for MySQL, Postgres, MongoDB (Connectors)
- **Reporting**: ReportLab (PDF Generation)
- **Security**: Regex-based IDS Engine & PyPDF2 Log Parsing

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- pip (Python Package Installer)

### 2. Clone & Install
```bash
# Navigate to the project directory
cd "c:/VIGNESHWARAN/Dsu/Sem 4/Database Management System-4/proj/db"

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Initialization
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Create Administrator
```bash
python manage.py createsuperuser
```

### 5. Launch the Platform
```bash
python manage.py runserver
```
Access the dashboard at: `http://127.0.0.1:8000/`

---

## 🔒 Security Policy
- **Read-Only Logs**: The platform enforces strict read-only access to system log tables, even via the interactive terminal, to prevent tampering.
- **Admin Verification**: All critical infrastructure tools (Server Console, API Engine, SQL Terminal) require full administrator privileges.

---

## 📜 License
This project is developed as part of the Database Management System-4 curriculum. All rights reserved.

---
**Developed by Antigravity AI**
