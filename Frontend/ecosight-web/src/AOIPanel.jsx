// AOIPanel.jsx — Click map to pin + auto-analyse all 6 fields
import { useEffect, useRef, useState } from 'react';
import { MapPin, Download, Trash2, RefreshCw, Mail, Satellite, Info, Loader, TreePine, Users, CloudRain, Wheat, Droplets, Activity } from 'lucide-react';

const ALERT_META = {
  green:  { color: '#10b981', bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.3)',  label: '🟢 Normal'   },
  yellow: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.3)',  label: '🟡 Heads Up' },
  orange: { color: '#f97316', bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.3)',  label: '🟠 Warning'  },
  red:    { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)',   label: '🔴 Critical' },
};

const FIELD_META = [
  { key:'weather',    icon:<CloudRain size={13}/>,  label:'Weather',    color:'#8b5cf6' },
  { key:'ndvi',       icon:<Activity size={13}/>,   label:'NDVI/NDWI',  color:'#10b981' },
  { key:'forest',     icon:<TreePine size={13}/>,   label:'Forest',     color:'#22c55e' },
  { key:'crop',       icon:<Wheat size={13}/>,      label:'Crop',       color:'#f59e0b' },
  { key:'drought',    icon:<Droplets size={13}/>,   label:'Drought',    color:'#ef4444' },
  { key:'population', icon:<Users size={13}/>,      label:'Population', color:'#3b82f6' },
];

export default function AOIPanel({ currentUser, onAlertTriggered, theme }) {
  const mapRef    = useRef(null);
  const leafMap   = useRef(null);
  const drawnLyr  = useRef(null);
  const pinMkr    = useRef(null);

  const [aois,          setAois]          = useState([]);
  const [selected,      setSelected]      = useState(null);
  const [analysis,      setAnalysis]      = useState(null);
  const [fieldData,     setFieldData]     = useState({});
  const [loading,       setLoading]       = useState(false);
  const [loadingFields, setLoadingFields] = useState({});
  const [emailInput,    setEmailInput]    = useState(currentUser?.email || '');
  const [emailStatus,   setEmailStatus]   = useState(null);
  const [mapReady,      setMapReady]      = useState(false);
  const [activeTab,     setActiveTab]     = useState('indices');

  useEffect(() => {
    const init = () => {
      if (!window.L || !mapRef.current || leafMap.current) return;
      const map = window.L.map(mapRef.current, { center:[20.5937,78.9629], zoom:5 });
      applyTile(map, theme);
      const drawn = new window.L.FeatureGroup();
      map.addLayer(drawn); drawnLyr.current = drawn;
      const ctrl = new window.L.Control.Draw({
        edit: { featureGroup: drawn },
        draw: {
          polygon:   { shapeOptions:{ color:'#00d4aa', fillOpacity:0.08, weight:2 } },
          rectangle: { shapeOptions:{ color:'#00d4aa', fillOpacity:0.08, weight:2 } },
          circle:false, circlemarker:false, polyline:false, marker:false,
        },
      });
      map.addControl(ctrl);
      map.on('click', e => dropPin(map, e.latlng.lat, e.latlng.lng));
      map.on(window.L.Draw.Event.CREATED, e => {
        drawn.clearLayers(); drawn.addLayer(e.layer);
        const b = e.layer.getBounds(), c = b.getCenter();
        createAOI(c.lat, c.lng, { n:b.getNorth(), s:b.getSouth(), e:b.getEast(), w:b.getWest() });
      });
      map.on(window.L.Draw.Event.DELETED, () => {
        setAnalysis(null); setSelected(null); setFieldData({});
        if (pinMkr.current) { map.removeLayer(pinMkr.current); pinMkr.current = null; }
      });
      leafMap.current = map; setMapReady(true);
    };
    if (window.L) init();
    else { const iv = setInterval(()=>{ if(window.L){clearInterval(iv);init();} },200); return ()=>clearInterval(iv); }
    return () => { if(leafMap.current){leafMap.current.remove();leafMap.current=null;} };
  }, []);

  useEffect(() => {
    if (!leafMap.current||!window.L) return;
    leafMap.current.eachLayer(l=>{ if(l._url) leafMap.current.removeLayer(l); });
    applyTile(leafMap.current, theme);
    if (drawnLyr.current) leafMap.current.addLayer(drawnLyr.current);
  }, [theme]);

  function applyTile(map, t) {
    window.L.tileLayer(
      t==='dark-theme' ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                       : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      { attribution:'© OSM © CARTO', maxZoom:19 }
    ).addTo(map);
  }

  function dropPin(map, lat, lng) {
    if (pinMkr.current) map.removeLayer(pinMkr.current);
    const icon = window.L.divIcon({
      className:'',
      html:`<div style="width:24px;height:24px;background:#00d4aa;border-radius:50% 50% 50% 0;transform:rotate(-45deg);border:3px solid #fff;box-shadow:0 2px 12px rgba(0,212,170,0.7)"></div>`,
      iconSize:[24,24], iconAnchor:[12,24],
    });
    pinMkr.current = window.L.marker([lat,lng],{icon}).addTo(map);
    createAOI(lat, lng, { n:lat+0.05, s:lat-0.05, e:lng+0.05, w:lng-0.05 });
  }

  function createAOI(lat, lng, bounds) {
    const aoi = {
      id: Date.now(),
      name: `AOI-${new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'})}`,
      bounds, center:{ lat:+lat.toFixed(5), lng:+lng.toFixed(5) },
      alertLevel:null, createdAt:new Date().toISOString(),
    };
    setAois(p=>[aoi,...p.slice(0,4)]);
    setSelected(aoi); setFieldData({}); setActiveTab('indices');
    runAnalysis(aoi);
  }

  const runAnalysis = async (aoi) => {
    setLoading(true); setAnalysis(null); setEmailStatus(null);
    try {
      const res = await fetch('http://127.0.0.1:8000/aoi/analyze',{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ lat:aoi.center.lat, lng:aoi.center.lng,
          north:aoi.bounds.n, south:aoi.bounds.s, east:aoi.bounds.e, west:aoi.bounds.w,
          aoi_name:aoi.name, user_email:emailInput||null }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setAnalysis(data);
      const upd = {...aoi, alertLevel:data.alert_level};
      setAois(p=>p.map(a=>a.id===aoi.id?upd:a)); setSelected(upd);
      if (data.alert_level !== 'green') {
        onAlertTriggered?.({ id:Date.now(), aoi:upd, level:data.alert_level,
          summary:data.summary, timestamp:data.timestamp,
          indices:{ndvi:data.ndvi, ndwi:data.ndwi, ndbi:data.ndbi} });
      }
    } catch {
      const demo = { ndvi:0.38, ndwi:-0.14, ndbi:0.09, temperature:31, wind_speed:22,
        precipitation:4.2, alert_level:'yellow',
        summary:'🟡 Demo values — backend offline.', timestamp:new Date().toISOString(),
        data_sources:['Demo (backend offline)'] };
      setAnalysis(demo);
      const upd = {...aoi, alertLevel:'yellow'};
      setAois(p=>p.map(a=>a.id===aoi.id?upd:a)); setSelected(upd);
    } finally { setLoading(false); }
    runFields(aoi);
  };

  const runFields = async (aoi) => {
    const lat = aoi.center.lat.toFixed(4), lng = aoi.center.lng.toFixed(4);
    let city = 'India';
    try {
      const nom = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
        {headers:{'Accept-Language':'en','User-Agent':'GeoDrishti/1.0'}}).then(r=>r.json());
      city = nom?.address?.state || nom?.address?.city || nom?.address?.county || 'India';
    } catch {}
    const queries = {
      weather:`weather in ${city}`, forest:`forest cover in ${city}`,
      crop:`crop yield for ${city}`, drought:'drought index analysis',
      population:'population data for India 2017', ndvi:`vegetation ndvi for ${city}`,
    };
    setLoadingFields(Object.fromEntries(Object.keys(queries).map(k=>[k,true])));
    await Promise.allSettled(Object.entries(queries).map(async ([field, query]) => {
      try {
        const res = await fetch('http://127.0.0.1:8000/chat',{
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({message:query}),
        });
        const d = await res.json();
        setFieldData(p=>({...p,[field]:d.reply}));
      } catch {
        setFieldData(p=>({...p,[field]:'⚠️ Backend offline.'}));
      } finally {
        setLoadingFields(p=>({...p,[field]:false}));
      }
    }));
  };

  const sendEmail = async () => {
    if (!emailInput||!analysis||!selected) return;
    setEmailStatus('sending');
    try {
      const res = await fetch('http://127.0.0.1:8000/alert/email',{
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ to_email:emailInput, aoi_name:selected.name,
          level:analysis.alert_level, summary:analysis.summary,
          lat:selected.center.lat, lng:selected.center.lng }),
      });
      const d = await res.json();
      setEmailStatus(d.success?'sent':'failed');
    } catch { setEmailStatus('failed'); }
  };

  const exportGeoJSON = () => {
    if (!selected||!analysis) return;
    const geojson = { type:'FeatureCollection', generator:'ISRO GeoDrishti EcoSight',
      generated_at:new Date().toISOString(),
      features:[{ type:'Feature',
        geometry:{ type:'Polygon', coordinates:[[
          [selected.bounds.w,selected.bounds.n],[selected.bounds.e,selected.bounds.n],
          [selected.bounds.e,selected.bounds.s],[selected.bounds.w,selected.bounds.s],
          [selected.bounds.w,selected.bounds.n],
        ]]},
        properties:{ name:selected.name, center_lat:selected.center.lat, center_lng:selected.center.lng,
          ndvi:analysis.ndvi, ndwi:analysis.ndwi, ndbi:analysis.ndbi,
          temperature_c:analysis.temperature, wind_kmh:analysis.wind_speed, precip_mm:analysis.precipitation,
          alert_level:analysis.alert_level, summary:analysis.summary,
          field_analysis:fieldData, data_sources:analysis.data_sources,
          exported_at:new Date().toISOString() },
      }],
    };
    const blob = new Blob([JSON.stringify(geojson,null,2)],{type:'application/json'});
    const a = Object.assign(document.createElement('a'),{
      href:URL.createObjectURL(blob), download:`${selected.name}_geodrishti_full.geojson` });
    a.click(); URL.revokeObjectURL(a.href);
  };

  const renderText = (text) => {
    if (!text) return null;
    return text.split(/(\*\*[^*]+\*\*)/g).map((p,i)=>
      p.startsWith('**')&&p.endsWith('**') ? <strong key={i}>{p.slice(2,-2)}</strong> : <span key={i}>{p}</span>
    );
  };

  const am = analysis ? ALERT_META[analysis.alert_level]||ALERT_META.green : null;

  return (
    <div className="aoi-panel">
      <div className="aoi-map-wrap">
        <div className="aoi-map-header">
          <MapPin size={14} style={{color:'var(--accent)'}}/>
          <span>Click anywhere on map to pin &amp; analyse</span>
          <span className="aoi-tip">or draw a region with the toolbar</span>
        </div>
        <div ref={mapRef} className="aoi-map" role="application" aria-label="Interactive map"/>
        {!mapReady && <div className="aoi-map-loading"><Loader size={22} className="spin"/><span>Loading map…</span></div>}
      </div>

      <div className="aoi-sidebar">
        {aois.length > 0 && (
          <div className="aoi-history">
            <div className="aoi-section-label">Recent Pins</div>
            {aois.map(a => {
              const m = ALERT_META[a.alertLevel||'green'];
              return (
                <button key={a.id} className={`aoi-history-item ${selected?.id===a.id?'active':''}`}
                        style={selected?.id===a.id?{borderColor:m.color}:{}}
                        onClick={()=>{setSelected(a);runAnalysis(a);}}>
                  <span style={{color:m.color}}>{m.label.split(' ')[0]}</span>
                  <span className="aoi-history-name">{a.name}</span>
                  <span className="aoi-history-coords">{a.center.lat}, {a.center.lng}</span>
                </button>
              );
            })}
          </div>
        )}

        {aois.length === 0 && (
          <div className="aoi-empty">
            <Satellite size={36} style={{color:'var(--text-dim)',marginBottom:10}}/>
            <p>Click anywhere on the map to pin a location.</p>
            <p className="aoi-empty-sub">All 6 data fields auto-load instantly.</p>
          </div>
        )}

        {loading && <div className="aoi-loading"><Loader size={16} className="spin"/><span>Running satellite analysis…</span></div>}

        {analysis && !loading && (
          <>
            <div className="aoi-alert-banner" style={{background:am.bg,borderColor:am.border}}>
              <div className="aoi-alert-level" style={{color:am.color}}>{am.label}</div>
              <div className="aoi-alert-desc">{analysis.summary}</div>
            </div>

            <div className="aoi-tabs">
              <button className={`aoi-tab ${activeTab==='indices'?'active':''}`} onClick={()=>setActiveTab('indices')}>Indices</button>
              <button className={`aoi-tab ${activeTab==='predictions'?'active':''}`} onClick={()=>setActiveTab('predictions')}>Predictions</button>
              <button className={`aoi-tab ${activeTab==='fields'?'active':''}`} onClick={()=>setActiveTab('fields')}>
                All Fields {Object.values(loadingFields).some(Boolean) && <Loader size={10} className="spin" style={{marginLeft:4}}/>}
              </button>
            </div>

            {activeTab==='indices' && (
              <>
                <div className="index-grid">
                  {[
                    {key:'ndvi',label:'NDVI',desc:'Vegetation',color:'#10b981',interpret:v=>v>0.6?'Dense':v>0.3?'Moderate':v>0.1?'Sparse':'Bare'},
                    {key:'ndwi',label:'NDWI',desc:'Water',    color:'#3b82f6',interpret:v=>v>0.2?'Water body':v>0?'Moist':'Dry'},
                    {key:'ndbi',label:'NDBI',desc:'Built-up', color:'#f59e0b',interpret:v=>v>0.2?'High urban':v>0?'Moderate':'Low'},
                  ].map(({key,label,desc,color,interpret})=>(
                    <div key={key} className="index-card" style={{borderTopColor:color}}>
                      <div className="index-label" style={{color}}>{label}</div>
                      <div className="index-value">{analysis[key]??'N/A'}</div>
                      <div className="index-desc">{desc}</div>
                      {analysis[key]!=null && <div className="index-interpret" style={{color}}>{interpret(analysis[key])}</div>}
                    </div>
                  ))}
                </div>
                <div className="aoi-section-label" style={{marginTop:10}}>Live Conditions</div>
                <div className="aoi-weather-row">
                  {[{label:'Temp',value:`${analysis.temperature}°C`},{label:'Wind',value:`${analysis.wind_speed} km/h`},{label:'Precip',value:`${analysis.precipitation} mm`}].map(w=>(
                    <div key={w.label} className="aoi-weather-chip">
                      <span className="weather-chip-label">{w.label}</span>
                      <span className="weather-chip-value">{w.value}</span>
                    </div>
                  ))}
                </div>
                <div className="aoi-source-note"><Info size={10}/>{(analysis.data_sources||[]).join(' · ')}</div>
              </>
            )}

            {activeTab==='predictions' && (
              <div className="predictions-list">
                {/* Crop Predictions */}
                {analysis.predictions?.crop && !analysis.predictions.crop.error && (
                  <div className="prediction-block" style={{borderLeftColor:'#f59e0b'}}>
                    <div className="prediction-header" style={{color:'#f59e0b'}}>
                      <Wheat size={13}/> Crop Yield Prediction
                    </div>
                    <div className="prediction-content">
                      {analysis.predictions.crop.predictions?.slice(0,3).map((p,i)=>(
                        <div key={i} className="crop-pred-item">
                          <span className="crop-name">{p.crop}</span>
                          <span className="crop-conf">{p.confidence}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Forest Predictions */}
                {analysis.predictions?.forest && !analysis.predictions.forest.error && (
                  <div className="prediction-block" style={{borderLeftColor:'#10b981'}}>
                    <div className="prediction-header" style={{color:'#10b981'}}>
                      <TreePine size={13}/> Forest Health Prediction
                    </div>
                    <div className="prediction-content">
                      <div className="forest-metrics">
                        <div>Alert Level: {['No Alert','Mild','Severe','Critical'][analysis.predictions.forest.alert_level] || 'N/A'}</div>
                        <div>Future NDVI: {analysis.predictions.forest.future_ndvi}</div>
                        <div>Future Cover: {analysis.predictions.forest.future_cover_sqkm} km²</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Drought Predictions */}
                {analysis.predictions?.drought && !analysis.predictions.drought.error && (
                  <div className="prediction-block" style={{borderLeftColor:'#ef4444'}}>
                    <div className="prediction-header" style={{color:'#ef4444'}}>
                      <Droplets size={13}/> Drought Risk Prediction
                    </div>
                    <div className="prediction-content">
                      <div className="drought-metrics">
                        <div>Category: {analysis.predictions.drought.category}</div>
                        <div>Status: {analysis.predictions.drought.status ? 'Drought' : 'Normal'}</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Population Predictions */}
                {analysis.predictions?.population && !analysis.predictions.population.error && (
                  <div className="prediction-block" style={{borderLeftColor:'#3b82f6'}}>
                    <div className="prediction-header" style={{color:'#3b82f6'}}>
                      <Users size={13}/> Urbanization Prediction
                    </div>
                    <div className="prediction-content">
                      <div className="pop-metrics">
                        <div>Future Pop: {analysis.predictions.population.future_population_millions}M</div>
                        <div>Urban Rate: {analysis.predictions.population.future_urbanization_rate}%</div>
                        <div>Growth Rate: {analysis.predictions.population.growth_rate}%</div>
                      </div>
                    </div>
                  </div>
                )}

                {(!analysis.predictions || Object.values(analysis.predictions).every(p=>p.error)) && (
                  <div className="prediction-error">
                    <Info size={14} style={{color:'var(--text-dim)'}}/>
                    ML models not loaded or prediction failed.
                  </div>
                )}
              </div>
            )}

            {activeTab==='fields' && (
              <div className="fields-list">
                {FIELD_META.map(({key,icon,label,color})=>(
                  <div key={key} className="field-block" style={{borderLeftColor:color}}>
                    <div className="field-block-header" style={{color}}>
                      {icon}{label}
                      {loadingFields[key] && <Loader size={10} className="spin" style={{marginLeft:'auto'}}/>}
                    </div>
                    <div className="field-block-content">
                      {loadingFields[key] ? <span style={{color:'var(--text-dim)',fontSize:'0.73rem'}}>Fetching…</span>
                       : fieldData[key] ? fieldData[key].split('\n').map((line,i,arr)=>(
                           <span key={i}>{renderText(line)}{i<arr.length-1&&<br/>}</span>
                         ))
                       : <span style={{color:'var(--text-dim)',fontSize:'0.73rem'}}>Pending…</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="aoi-email-row">
              <input className="aoi-email-input" type="email" placeholder="your@email.com for alerts"
                     value={emailInput} onChange={e=>setEmailInput(e.target.value)}/>
              <button className="aoi-email-btn" onClick={sendEmail} disabled={emailStatus==='sending'||!emailInput}>
                {emailStatus==='sending'?<Loader size={13} className="spin"/>:<Mail size={13}/>}
              </button>
            </div>
            {emailStatus==='sent'   && <div className="email-status ok">✓ Alert email sent!</div>}
            {emailStatus==='failed' && <div className="email-status err">✗ Failed — check SMTP config.</div>}

            <div className="aoi-actions">
              <button className="aoi-btn primary" onClick={()=>selected&&runAnalysis(selected)} disabled={loading}>
                <RefreshCw size={12}/> Re-analyse
              </button>
              <button className="aoi-btn secondary" onClick={exportGeoJSON}><Download size={12}/> GeoJSON</button>
              <button className="aoi-btn danger" onClick={()=>{
                drawnLyr.current?.clearLayers();
                if(pinMkr.current&&leafMap.current){leafMap.current.removeLayer(pinMkr.current);pinMkr.current=null;}
                setSelected(null);setAnalysis(null);setFieldData({});setAois([]);
              }}><Trash2 size={12}/> Clear</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}