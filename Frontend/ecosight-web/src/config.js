// src/config.js
// Central place to configure your backend URL.
// In production (Vercel), this points to your HF Space.
// Locally, it points to uvicorn on port 8000.

const IS_PROD = import.meta.env.PROD; // true on Vercel build, false on npm run dev

export const API_BASE = IS_PROD
  ? "https://geodrishti-geodrishti.hf.space"   // ← your HF Space URL
  : "http://127.0.0.1:8000";

// Usage in any component:
//   import { API_BASE } from './config';
//   fetch(`${API_BASE}/chat`, { ... })