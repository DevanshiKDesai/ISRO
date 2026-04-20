const IS_PROD = import.meta.env.PROD;

export const API_BASE = import.meta.env.VITE_API_BASE || (
  IS_PROD ? 'https://geodrishti-geodrishti.hf.space' : 'http://127.0.0.1:8000'
);

export const POWER_BI_EMBEDS = {
  forest: import.meta.env.VITE_POWERBI_FOREST || '',
  population: import.meta.env.VITE_POWERBI_POPULATION || '',
  weather: import.meta.env.VITE_POWERBI_WEATHER || '',
  crop: import.meta.env.VITE_POWERBI_CROP || '',
  drought: import.meta.env.VITE_POWERBI_DROUGHT || '',
};
