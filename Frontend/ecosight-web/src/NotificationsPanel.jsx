// NotificationsPanel.jsx — Alert history, email dispatch, PDF-style report download
import { useState } from 'react';
import {
  Bell, AlertTriangle, CheckCircle, AlertCircle,
  Download, Mail, Trash2, Clock, MapPin, X, Filter
} from 'lucide-react';

const LEVEL_META = {
  green:  { color: '#10b981', bg: 'rgba(16,185,129,0.1)',  border: 'rgba(16,185,129,0.25)', icon: <CheckCircle size={14}/>,  label: 'Normal'    },
  yellow: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.25)', icon: <Bell size={14}/>,         label: 'Heads Up'  },
  orange: { color: '#f97316', bg: 'rgba(249,115,22,0.1)',  border: 'rgba(249,115,22,0.25)', icon: <AlertTriangle size={14}/>, label: 'Warning'   },
  red:    { color: '#ef4444', bg: 'rgba(239,68,68,0.1)',   border: 'rgba(239,68,68,0.25)',  icon: <AlertCircle size={14}/>,  label: 'Critical'  },
};

export default function NotificationsPanel({ alerts, onDismiss, onClearAll }) {
  const [filter, setFilter]     = useState('all'); // all | yellow | orange | red
  const [emailMap, setEmailMap] = useState({});    // alertId -> email input
  const [sentMap, setSentMap]   = useState({});    // alertId -> bool

  const filtered = alerts.filter(a => filter === 'all' || a.level === filter);

  const sendEmail = async (alert) => {
    const email = emailMap[alert.id];
    if (!email) return;
    try {
      const res = await fetch('http://127.0.0.1:8000/alert/email', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          to_email: email,
          aoi_name: alert.aoi?.name || 'AOI',
          level:    alert.level,
          summary:  alert.summary,
          lat:      alert.aoi?.center?.lat || 0,
          lng:      alert.aoi?.center?.lng || 0,
        }),
      });
      const data = await res.json();
      setSentMap(p => ({ ...p, [alert.id]: data.success }));
    } catch {
      setSentMap(p => ({ ...p, [alert.id]: false }));
    }
  };

  const downloadReport = (alert) => {
    const meta  = LEVEL_META[alert.level] || LEVEL_META.green;
    const lines = [
      '='.repeat(60),
      '  GEODRISHTI ECOSIGHT — ENVIRONMENTAL ALERT REPORT',
      '  ISRO EcoSight Monitoring System',
      '='.repeat(60),
      '',
      `ALERT LEVEL  : ${meta.label.toUpperCase()} (${alert.level})`,
      `AOI NAME     : ${alert.aoi?.name || 'N/A'}`,
      `COORDINATES  : ${alert.aoi?.center?.lat || 'N/A'}, ${alert.aoi?.center?.lng || 'N/A'}`,
      `GENERATED AT : ${new Date(alert.timestamp).toLocaleString('en-IN')}`,
      '',
      'SUMMARY',
      '-'.repeat(40),
      alert.summary,
      '',
      'SATELLITE INDICES',
      '-'.repeat(40),
      `NDVI (Vegetation)  : ${alert.indices?.ndvi ?? 'N/A'}`,
      `NDWI (Water)       : ${alert.indices?.ndwi ?? 'N/A'}`,
      `NDBI (Built-up)    : ${alert.indices?.ndbi ?? 'N/A'}`,
      '',
      'DATA SOURCES',
      '-'.repeat(40),
      'Open-Meteo API (live weather)',
      'NASA POWER API (solar/precipitation composites)',
      'GeoDrishti offline datasets (forest, crop, drought, population)',
      '',
      '='.repeat(60),
      'This report was generated automatically by GeoDrishti EcoSight.',
      'For ground verification, contact ISRO EcoSight team.',
      '='.repeat(60),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const a    = Object.assign(document.createElement('a'), {
      href:     URL.createObjectURL(blob),
      download: `geodrishti_alert_${alert.aoi?.name || 'report'}_${Date.now()}.txt`,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="notif-panel">
      {/* Header */}
      <div className="notif-header">
        <div className="notif-title">
          <Bell size={16} style={{ color: 'var(--accent)' }} />
          Alert History
          {alerts.length > 0 && (
            <span className="notif-count">{alerts.length}</span>
          )}
        </div>
        <div className="notif-header-actions">
          {alerts.length > 0 && (
            <button
              className="notif-clear-btn"
              onClick={onClearAll}
              aria-label="Clear all alerts"
            >
              <Trash2 size={13} /> Clear all
            </button>
          )}
        </div>
      </div>

      {/* Filter bar */}
      <div className="notif-filters" role="group" aria-label="Filter alerts by level">
        {['all', 'yellow', 'orange', 'red'].map(f => (
          <button
            key={f}
            className={`notif-filter-btn ${filter === f ? 'active' : ''}`}
            style={filter === f && f !== 'all' ? { color: LEVEL_META[f]?.color, borderColor: LEVEL_META[f]?.color } : {}}
            onClick={() => setFilter(f)}
            aria-pressed={filter === f}
          >
            {f === 'all' ? 'All' : LEVEL_META[f].label}
          </button>
        ))}
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <div className="notif-empty">
          <CheckCircle size={36} style={{ color: 'var(--accent)', marginBottom: 10 }} />
          <p>{filter === 'all' ? 'No alerts yet. Draw an AOI to start monitoring.' : `No ${filter} alerts.`}</p>
        </div>
      )}

      {/* Alert list */}
      <div className="notif-list">
        {filtered.map(alert => {
          const meta = LEVEL_META[alert.level] || LEVEL_META.green;
          return (
            <div
              key={alert.id}
              className="notif-item"
              style={{ borderLeftColor: meta.color, background: meta.bg }}
              role="article"
              aria-label={`${meta.label} alert for ${alert.aoi?.name}`}
            >
              {/* Top row */}
              <div className="notif-item-top">
                <span className="notif-level" style={{ color: meta.color }}>
                  {meta.icon} {meta.label}
                </span>
                <span className="notif-time">
                  <Clock size={11} />
                  {new Date(alert.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
                </span>
                <button
                  className="notif-dismiss"
                  onClick={() => onDismiss(alert.id)}
                  aria-label="Dismiss alert"
                >
                  <X size={13} />
                </button>
              </div>

              {/* AOI info */}
              <div className="notif-aoi-name">
                <MapPin size={11} style={{ color: 'var(--text-dim)' }} />
                {alert.aoi?.name || 'Unknown AOI'}
                {alert.aoi?.center && (
                  <span className="notif-coords">
                    {alert.aoi.center.lat}, {alert.aoi.center.lng}
                  </span>
                )}
              </div>

              {/* Summary */}
              <p className="notif-summary">{alert.summary}</p>

              {/* Indices pills */}
              {alert.indices && (
                <div className="notif-indices">
                  {Object.entries(alert.indices).map(([k, v]) => (
                    <span key={k} className="notif-index-pill">
                      {k.toUpperCase()}: {v ?? 'N/A'}
                    </span>
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="notif-actions">
                <button
                  className="notif-action-btn"
                  onClick={() => downloadReport(alert)}
                  aria-label="Download report"
                  title="Download .txt report"
                >
                  <Download size={12} /> Report
                </button>

                <div className="notif-email-row">
                  <input
                    className="notif-email-input"
                    type="email"
                    placeholder="Email alert…"
                    value={emailMap[alert.id] || ''}
                    onChange={e => setEmailMap(p => ({ ...p, [alert.id]: e.target.value }))}
                    aria-label="Recipient email for this alert"
                  />
                  <button
                    className="notif-action-btn accent"
                    onClick={() => sendEmail(alert)}
                    disabled={!emailMap[alert.id]}
                    aria-label="Send email"
                  >
                    <Mail size={12} />
                    {sentMap[alert.id] === true  ? '✓ Sent' :
                     sentMap[alert.id] === false ? '✗ Failed' : 'Send'}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}