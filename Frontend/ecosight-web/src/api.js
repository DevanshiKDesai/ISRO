import { API_BASE } from './config';

async function parseResponse(response, mode = 'json') {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return mode === 'text' ? response.text() : response.json();
}

export async function analyzeAOI(payload, exportMode = 'json') {
  const response = await fetch(`${API_BASE}/aoi/analyze?export=${exportMode}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return parseResponse(response, exportMode === 'csv' ? 'text' : 'json');
}

export async function sendAlertEmail(payload) {
  const response = await fetch(`${API_BASE}/tool/alert/email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function askChat(message) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  return parseResponse(response);
}
