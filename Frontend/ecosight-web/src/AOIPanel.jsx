import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CloudRain,
  Download,
  Droplets,
  Info,
  Loader,
  Mail,
  MapPin,
  RefreshCw,
  Satellite,
  Trash2,
  TreePine,
  Users,
  Wheat,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2,
  Activity,
  Zap,
  ShieldAlert,
  Wind,
  Thermometer,
  Layers,
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  RadialBarChart, 
  RadialBar, 
  Tooltip as RechartsTooltip,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';
import { analyzeAOI, sendAlertEmail } from './api';

// Utility for tailwind classes
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const ALERT_META = {
  GREEN: { color: '#10b981', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-500', label: 'Safe' },
  YELLOW: { color: '#f59e0b', bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-500', label: 'Warning' },
  RED: { color: '#ef4444', bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-500', label: 'Danger' },
};

const PARAM_META = {
  crop: { label: 'Crop Prediction', icon: <Wheat size={16} />, desc: 'AI-recommended crop based on soil and weather.', color: '#f59e0b' },
  drought: { label: 'Drought Status', icon: <Droplets size={16} />, desc: 'Current drought severity and active risk level.', color: '#ef4444' },
  disaster: { label: 'Disaster Risk', icon: <CloudRain size={16} />, desc: 'Probability and intensity of extreme weather events.', color: '#8b5cf6' },
  forest: { label: 'Forest Health', icon: <TreePine size={16} />, desc: 'Deforestation alerts and future vegetation cover.', color: '#10b981' },
  population: { label: 'Urban Growth', icon: <Users size={16} />, desc: '5-year population and infrastructure projections.', color: '#3b82f6' },
};

function PredictionCard({ id, value, loading, subValue }) {
  const meta = PARAM_META[id] || { label: id, icon: <Activity size={16} />, color: '#6366f1' };
  
  // Status logic for ML predictions
  let status = 'Normal';
  let statusColor = 'text-emerald-500';
  let bgColor = 'bg-emerald-500/10';
  
  if (loading) {
    status = 'Analyzing';
    statusColor = 'text-amber-500';
    bgColor = 'bg-amber-500/10';
  } else if (!value || value === 'N/A') {
    status = 'No Data';
    statusColor = 'text-gray-400';
    bgColor = 'bg-gray-400/10';
  } else {
    // Custom logic per domain
    const valStr = String(value).toLowerCase();
    if (valStr.includes('risk') || valStr.includes('alert') || valStr.includes('severe') || valStr.includes('high')) {
      status = 'Warning';
      statusColor = 'text-rose-500';
      bgColor = 'bg-rose-500/10';
    } else if (valStr.includes('moderate')) {
      status = 'Monitor';
      statusColor = 'text-amber-500';
      bgColor = 'bg-amber-500/10';
    }
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="group relative flex flex-col gap-3 rounded-2xl border border-white/5 bg-white/5 p-4 backdrop-blur-md transition-all hover:bg-white/10 hover:shadow-xl hover:shadow-black/20"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/10 text-white/80 group-hover:text-white" style={{ color: meta.color }}>
            {meta.icon}
          </div>
          <span className="text-sm font-medium text-white/60">{meta.label}</span>
        </div>
        <div className={cn("rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider", bgColor, statusColor)}>
          {status}
        </div>
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex items-baseline gap-2 overflow-hidden">
          <span className="truncate text-xl font-bold tracking-tight text-white lg:text-2xl">
            {loading ? <span className="animate-pulse">...</span> : (value || 'N/A')}
          </span>
        </div>
        {subValue && !loading && (
          <span className="text-[10px] font-medium text-white/40 uppercase tracking-wide">
            {subValue}
          </span>
        )}
      </div>

      {/* Tooltip on hover */}
      <div className="pointer-events-none absolute -top-12 left-0 z-50 w-full opacity-0 transition-opacity group-hover:opacity-100">
        <div className="rounded-lg bg-black/90 p-2 text-[10px] leading-relaxed text-white/90 shadow-2xl backdrop-blur-md border border-white/10">
          {meta.desc}
        </div>
      </div>
    </motion.div>
  );
}

export default function AOIPanel({ currentUser, onAlertTriggered, theme }) {
  const mapRef = useRef(null);
  const leafMap = useRef(null);
  const drawnLyr = useRef(null);
  const pinMkr = useRef(null);

  const [aois, setAois] = useState([]);
  const [selected, setSelected] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [emailInput, setEmailInput] = useState(currentUser?.email || '');
  const [emailStatus, setEmailStatus] = useState(null);
  const [mapReady, setMapReady] = useState(false);

  useEffect(() => {
    const init = () => {
      if (!window.L || !mapRef.current || leafMap.current) return;
      const map = window.L.map(mapRef.current, { center: [20.5937, 78.9629], zoom: 5 });
      applyTile(map, theme);
      const drawn = new window.L.FeatureGroup();
      map.addLayer(drawn);
      drawnLyr.current = drawn;
      
      const ctrl = new window.L.Control.Draw({
        edit: { featureGroup: drawn },
        draw: {
          polygon: { shapeOptions: { color: '#00d4aa', fillOpacity: 0.1, weight: 2 } },
          rectangle: { shapeOptions: { color: '#00d4aa', fillOpacity: 0.1, weight: 2 } },
          circle: false,
          circlemarker: false,
          polyline: false,
          marker: false,
        },
      });
      map.addControl(ctrl);

      map.on('click', (e) => dropPin(map, e.latlng.lat, e.latlng.lng));
      map.on(window.L.Draw.Event.CREATED, (e) => {
        drawn.clearLayers();
        drawn.addLayer(e.layer);
        const bounds = e.layer.getBounds();
        createAOI(bounds.getCenter().lat, bounds.getCenter().lng, {
          n: bounds.getNorth(), s: bounds.getSouth(), e: bounds.getEast(), w: bounds.getWest(),
        });
      });
      
      leafMap.current = map;
      setMapReady(true);
    };

    if (window.L) init();
    return () => { if (leafMap.current) { leafMap.current.remove(); leafMap.current = null; } };
  }, []);

  function applyTile(map, currentTheme) {
    window.L.tileLayer(
      currentTheme === 'dark-theme'
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      { attribution: '© OSM © CARTO', maxZoom: 19 }
    ).addTo(map);
  }

  function dropPin(map, lat, lng) {
    if (pinMkr.current) map.removeLayer(pinMkr.current);
    const icon = window.L.divIcon({
      className: '',
      html: '<div class="w-6 h-6 bg-accent rounded-full border-4 border-white shadow-lg animate-bounce"></div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
    pinMkr.current = window.L.marker([lat, lng], { icon }).addTo(map);
    createAOI(lat, lng, { n: lat + 0.05, s: lat - 0.05, e: lng + 0.05, w: lng - 0.05 });
  }

  function createAOI(lat, lng, bounds) {
    const aoi = {
      id: Date.now(),
      name: `AOI-${new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}`,
      bounds,
      center: { lat: +lat.toFixed(5), lng: +lng.toFixed(5) },
      alertLevel: 'green',
      createdAt: new Date().toISOString(),
    };
    setAois(prev => [aoi, ...prev.slice(0, 4)]);
    setSelected(aoi);
    runAnalysis(aoi);
  }

  async function runAnalysis(aoi) {
    setLoading(true);
    setAnalysis(null);
    try {
      const data = await analyzeAOI({
        lat: aoi.center.lat, lng: aoi.center.lng,
        north: aoi.bounds.n, south: aoi.bounds.s,
        east: aoi.bounds.e, west: aoi.bounds.w,
        aoi_name: aoi.name, user_email: emailInput || null,
      });
      setAnalysis(data);
      const level = (data.alert_level || 'GREEN').toUpperCase();
      const updated = { ...aoi, alertLevel: level.toLowerCase() };
      setAois(prev => prev.map(item => item.id === aoi.id ? updated : item));
      setSelected(updated);
      
      if (level !== 'GREEN') {
        onAlertTriggered?.({
          id: Date.now(), aoi: updated, level: level.toLowerCase(),
          summary: data.summary, timestamp: new Date().toISOString(),
        });
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  function downloadCSV() {
    if (!analysis) return;
    const rows = [
      ["Parameter", "Value", "Status"],
      ["Location", analysis.location, ""],
      ["Crop Prediction", analysis.domains?.crop_intelligence?.best_crop?.recommended_crop || "N/A", ""],
      ["Drought Status", analysis.domains?.drought_monitoring?.category?.primary_category || "N/A", ""],
      ["Disaster Risk", analysis.domains?.weather_disaster?.event?.primary_event || "N/A", ""],
      ["Forest Health", analysis.domains?.forest_health?.alert?.deforestation_alert_label || "N/A", ""],
      ["Urban Growth", analysis.domains?.urban_growth?.population?.future_population_millions || "N/A", "Million"],
      ["Temperature", analysis.inputs?.temperature, "°C"],
      ["Wind Speed", analysis.inputs?.wind_speed, "km/h"],
      ["NDVI", analysis.indices?.ndvi, ""],
      ["NDWI", analysis.indices?.ndwi, ""],
      ["NDBI", analysis.indices?.ndbi, ""],
    ];
    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `EcoSight_Report_${analysis.location}.csv`);
    document.body.appendChild(link);
    link.click();
  }

  function downloadReport() {
    if (!analysis) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(analysis, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `EcoSight_Report_${analysis.location}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }

  const alertMeta = analysis ? ALERT_META[analysis.alert_level || 'GREEN'] : ALERT_META.GREEN;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-[#0a0a0c] text-white lg:flex-row">
      {/* ── MAP AREA (70-75%) ──────────────────────────────────── */}
      <div className={cn(
        "relative transition-all duration-500 ease-in-out",
        isExpanded ? "w-0 lg:w-0" : "h-[50vh] w-full lg:h-full lg:flex-1"
      )}>
        <div ref={mapRef} className="h-full w-full" />
        <div className="pointer-events-none absolute left-6 top-6 flex flex-col gap-2">
          <div className="pointer-events-auto flex items-center gap-3 rounded-full bg-black/40 px-4 py-2 backdrop-blur-xl border border-white/10 shadow-2xl">
            <MapPin size={16} className="text-accent" />
            <span className="text-xs font-semibold tracking-wide uppercase text-white/90">
              Interactive Geospatial Dashboard
            </span>
          </div>
        </div>

        {/* Floating Toggle for Desktop */}
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="pointer-events-auto absolute right-6 top-6 hidden items-center justify-center rounded-xl bg-black/40 p-3 text-white/80 backdrop-blur-xl border border-white/10 transition-all hover:bg-black/60 hover:text-white lg:flex"
        >
          {isExpanded ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
        </button>
      </div>

      {/* ── PREDICTION PANEL (25-30%) ──────────────────────────── */}
      <div className={cn(
        "flex h-[50vh] flex-col border-t border-white/10 bg-[#0f0f12]/80 backdrop-blur-2xl transition-all duration-500 ease-in-out lg:h-full lg:border-l lg:border-t-0",
        isExpanded ? "w-full lg:w-full" : "w-full lg:w-[400px]"
      )}>
        {/* Panel Header */}
        <div className="flex items-center justify-between border-b border-white/5 px-6 py-4">
          <div className="flex items-center gap-3">
            <Activity size={20} className="text-accent" />
            <h2 className="text-lg font-bold tracking-tight">EcoSight Analytics</h2>
          </div>
          <div className="flex items-center gap-2">
            <button className="rounded-lg bg-white/5 p-2 text-white/60 transition-colors hover:bg-white/10 hover:text-white lg:hidden">
              <ChevronRight size={18} className="rotate-90" />
            </button>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto px-6 py-6 scrollbar-hide">
          {!selected ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-white/5 text-white/20">
                <Satellite size={32} />
              </div>
              <h3 className="mb-2 font-semibold text-white/80">No AOI Selected</h3>
              <p className="max-w-[200px] text-xs leading-relaxed text-white/40">
                Click anywhere on the map to analyze satellite indices and environmental risk factors.
              </p>
            </div>
          ) : (
            <div className="space-x-0 space-y-6">
              {/* Alert Level Banner */}
              <AnimatePresence mode="wait">
                {analysis && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className={cn("flex flex-col gap-2 rounded-2xl border p-4", alertMeta.bg, alertMeta.border)}
                  >
                    <div className="flex items-center gap-2">
                      <ShieldAlert size={16} className={alertMeta.text} />
                      <span className={cn("text-xs font-black uppercase tracking-widest", alertMeta.text)}>
                        {alertMeta.label} Alert
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-white/80">{analysis.summary}</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Parameter Cards Grid */}
              <div className={cn(
                "grid gap-4",
                isExpanded ? "grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5" : "grid-cols-1"
              )}>
                <PredictionCard 
                  id="crop" 
                  value={analysis?.domains?.crop_intelligence?.best_crop?.recommended_crop} 
                  subValue={analysis?.domains?.crop_intelligence?.best_crop?.confidence ? `${(analysis.domains.crop_intelligence.best_crop.confidence * 100).toFixed(0)}% Confidence` : ""}
                  loading={loading} 
                />
                <PredictionCard 
                  id="drought" 
                  value={analysis?.domains?.drought_monitoring?.category?.primary_category} 
                  subValue={analysis?.domains?.drought_monitoring?.status?.is_drought ? "Active Warning" : "Stable"}
                  loading={loading} 
                />
                <PredictionCard 
                  id="disaster" 
                  value={analysis?.domains?.weather_disaster?.event?.primary_event} 
                  subValue={analysis?.domains?.weather_disaster?.intensity?.intensity_label ? `${analysis.domains.weather_disaster.intensity.intensity_label} Intensity` : ""}
                  loading={loading} 
                />
                <PredictionCard 
                  id="forest" 
                  value={analysis?.domains?.forest_health?.alert?.deforestation_alert_label} 
                  subValue={analysis?.domains?.forest_health?.future_ndvi?.future_ndvi_label ? `Future: ${analysis.domains.forest_health.future_ndvi.future_ndvi_label}` : ""}
                  loading={loading} 
                />
                <PredictionCard 
                  id="population" 
                  value={analysis?.domains?.urban_growth?.population?.future_population_millions ? `${analysis.domains.urban_growth.population.future_population_millions}M` : ""} 
                  subValue={analysis?.domains?.urban_growth?.growth?.future_growth_rate ? `${analysis.domains.urban_growth.growth.future_growth_rate} Growth` : ""}
                  loading={loading} 
                />
              </div>

              {/* Export Options */}
              {analysis && !loading && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <button 
                    onClick={downloadCSV}
                    className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-white/5 bg-white/5 px-4 py-3 text-xs font-bold uppercase tracking-wider text-white/80 transition-all hover:bg-white/10 hover:text-white"
                  >
                    <Download size={14} /> Download CSV
                  </button>
                  <button 
                    onClick={downloadReport}
                    className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-white/5 bg-white/5 px-4 py-3 text-xs font-bold uppercase tracking-wider text-white/80 transition-all hover:bg-white/10 hover:text-white"
                  >
                    <Satellite size={14} /> Export Report
                  </button>
                </motion.div>
              )}

              {/* Secondary Data Grid */}
              {analysis && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-white/5 bg-white/5 p-3">
                    <div className="mb-1 flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-white/40">
                      <Thermometer size={12} /> Temperature
                    </div>
                    <div className="text-xl font-bold">{analysis.inputs?.temperature}°C</div>
                  </div>
                  <div className="rounded-xl border border-white/5 bg-white/5 p-3">
                    <div className="mb-1 flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-white/40">
                      <Wind size={12} /> Wind Speed
                    </div>
                    <div className="text-xl font-bold">{analysis.inputs?.wind_speed} km/h</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Footer */}
        {selected && (
          <div className="border-t border-white/5 bg-black/20 px-6 py-4">
            <div className="flex gap-2">
              <button 
                onClick={() => runAnalysis(selected)}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-bold text-white transition-all hover:brightness-110 active:scale-95"
              >
                <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
                {loading ? "Analyzing..." : "Refresh Prediction"}
              </button>
              <button className="flex items-center justify-center rounded-xl bg-white/5 px-4 py-2.5 text-white/80 transition-all hover:bg-white/10">
                <Download size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
