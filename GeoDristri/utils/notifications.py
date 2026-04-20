import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any


SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM = os.getenv("ALERT_FROM", "geodrishti@isro.gov.in")


def send_alert_email(to_email: str, aoi_name: str, level: str, summary: str, coords: dict[str, Any]) -> bool:
    if not SMTP_USER or not SMTP_PASSWORD:
        return False
    color = {"YELLOW": "#f59e0b", "RED": "#ef4444", "GREEN": "#10b981"}.get(level, "#10b981")
    html = (
        "<html><body style=\"font-family:sans-serif;background:#060a14;color:#e8edf5;padding:32px;\">"
        f"<div style=\"max-width:520px;margin:auto;background:#0d1526;border-radius:12px;border:2px solid {color};padding:28px;\">"
        f"<h2 style=\"color:{color};margin-top:0;\">GeoDhrishti Alert - {level}</h2>"
        f"<p><strong>AOI:</strong> {aoi_name}</p>"
        f"<p><strong>Coordinates:</strong> {coords.get('lat')}, {coords.get('lng')}</p>"
        f"<p><strong>Summary:</strong><br>{summary}</p>"
        f"<p style=\"color:#6b7fa3;font-size:12px;margin-top:24px;\">Generated - {datetime.utcnow().isoformat()}Z</p>"
        "</div></body></html>"
    )
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[GeoDhrishti] {level} Alert - {aoi_name}"
        msg["From"] = ALERT_FROM
        msg["To"] = to_email
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(ALERT_FROM, to_email, msg.as_string())
        return True
    except Exception:
        return False

