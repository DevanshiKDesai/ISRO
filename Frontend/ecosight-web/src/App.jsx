// App.jsx — GeoDrishti EcoSight v4.0
// Changes: redesigned emergency panel, alert toast popup, all previous features intact

import { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Stars } from '@react-three/drei';
import {
  LayoutDashboard, TreePine, Users, CloudRain, Wheat, Droplets,
  Mic, MessageSquare, UserCircle, PhoneCall, Sun, Moon, Bell,
  Activity, Lock, Mail, Eye, EyeOff, LogOut, Satellite,
  AlertCircle, User, ChevronRight, Shield, MapPin, Send,
  PhoneOff, X, Phone, AlertTriangle, CheckCircle, ExternalLink
} from 'lucide-react';
import VapiModule from '@vapi-ai/web';
import AOIPanel from './AOIPanel';
import NotificationsPanel from './NotificationsPanel';
import './App.css';

const Vapi = VapiModule.default || VapiModule;
const vapi = new Vapi(import.meta.env.VITE_VAPI_API_KEY);

const STORAGE_KEY = 'geodrishti_user';
const MOCK_USERS  = [{ email:'demo@isro.gov.in', password:'ecosight123', name:'Demo Analyst', role:'ANALYST' }];

function authLogin(email, pw) {
  const u = MOCK_USERS.find(x => x.email.toLowerCase()===email.toLowerCase() && x.password===pw);
  if (!u) throw new Error('Invalid credentials. Try demo@isro.gov.in / ecosight123');
  const s = { email:u.email, name:u.name, role:u.role };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); return s;
}
function authSignup(name, email, pw) {
  if (pw.length<8) throw new Error('Password must be at least 8 characters.');
  if (!email.includes('@')) throw new Error('Enter a valid email address.');
  const s = { email, name, role:'VIEWER' };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); return s;
}
function authLogout() { localStorage.removeItem(STORAGE_KEY); }
function getStoredUser() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { return null; } }

// ── AUTH MODAL ─────────────────────────────────────────────────
function AuthModal({ onClose, onAuthSuccess }) {
  const [tab,setTab]         = useState('login');
  const [showPw,setShowPw]   = useState(false);
  const [loading,setLoading] = useState(false);
  const [error,setError]     = useState('');
  const [le,setLe] = useState(''); const [lp,setLp] = useState('');
  const [sn,setSn] = useState(''); const [se,setSe] = useState(''); const [sp,setSp] = useState('');
  const firstRef = useRef(null);
  useEffect(()=>{ firstRef.current?.focus(); setError(''); },[tab]);
  useEffect(()=>{
    const fn=e=>{ if(e.key==='Escape') onClose(); };
    window.addEventListener('keydown',fn); return ()=>window.removeEventListener('keydown',fn);
  },[onClose]);
  const handle = async(fn)=>{ setError(''); setLoading(true);
    try { await new Promise(r=>setTimeout(r,500)); onAuthSuccess(fn()); }
    catch(e) { setError(e.message); } finally { setLoading(false); } };
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="auth-title"
         onClick={e=>{ if(e.target===e.currentTarget) onClose(); }}>
      <div className="auth-modal">
        <button className="modal-close-btn" onClick={onClose} aria-label="Close">×</button>
        <div className="auth-modal-header">
          <div className="auth-modal-logo" aria-hidden="true"><Satellite size={26} color="#000"/></div>
          <h2 className="auth-modal-title" id="auth-title">GeoDrishti</h2>
          <p className="auth-modal-subtitle">ISRO EcoSight — Secure Access Portal</p>
        </div>
        <div className="auth-tabs" role="tablist">
          {['login','signup'].map(t=>(
            <button key={t} role="tab" className={`auth-tab ${tab===t?'active':''}`}
                    aria-selected={tab===t} onClick={()=>setTab(t)}>
              {t==='login'?'Sign In':'Create Account'}
            </button>
          ))}
        </div>
        <form className="auth-form" noValidate
              onSubmit={e=>{ e.preventDefault(); tab==='login'?handle(()=>authLogin(le,lp)):handle(()=>authSignup(sn,se,sp)); }}>
          {error && <div className="form-error" role="alert"><AlertCircle size={14}/>{error}</div>}
          {tab==='signup' && (
            <div className="form-group">
              <label className="form-label" htmlFor="su-name">Full Name</label>
              <div className="form-input-wrap">
                <User size={15} className="form-input-icon" aria-hidden="true"/>
                <input id="su-name" ref={firstRef} className="form-input" type="text"
                       placeholder="Dr. A. P. J. Abdul Kalam" value={sn} onChange={e=>setSn(e.target.value)} autoComplete="name" required/>
              </div>
            </div>
          )}
          <div className="form-group">
            <label className="form-label" htmlFor="f-email">Email</label>
            <div className="form-input-wrap">
              <Mail size={15} className="form-input-icon" aria-hidden="true"/>
              <input id="f-email" ref={tab==='login'?firstRef:null} className="form-input" type="email"
                     placeholder="you@isro.gov.in" value={tab==='login'?le:se}
                     onChange={e=>tab==='login'?setLe(e.target.value):setSe(e.target.value)} autoComplete="email" required/>
            </div>
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="f-pw">Password</label>
            <div className="form-input-wrap">
              <Lock size={15} className="form-input-icon" aria-hidden="true"/>
              <input id="f-pw" className="form-input" type={showPw?'text':'password'}
                     placeholder={tab==='login'?'••••••••':'Min. 8 characters'} value={tab==='login'?lp:sp}
                     onChange={e=>tab==='login'?setLp(e.target.value):setSp(e.target.value)}
                     autoComplete={tab==='login'?'current-password':'new-password'} required minLength={tab==='signup'?8:undefined}/>
              <button type="button" className="password-toggle" onClick={()=>setShowPw(v=>!v)}
                      aria-label={showPw?'Hide':'Show'}>{showPw?<EyeOff size={14}/>:<Eye size={14}/>}</button>
            </div>
          </div>
          <button type="submit" className="auth-submit" disabled={loading} aria-busy={loading}>
            {loading?<span className="spinner" aria-hidden="true"/>:<Shield size={16}/>}
            {loading?'Authenticating…':tab==='login'?'Sign In':'Create Account'}
          </button>
          <p className="auth-footer-text">
            {tab==='login'?<>No account? <button type="button" onClick={()=>setTab('signup')}>Create one</button></>
                          :<>Have account? <button type="button" onClick={()=>setTab('login')}>Sign in</button></>}
          </p>
          {tab==='login' && <p className="auth-footer-text" style={{marginTop:4}}>Demo: <code style={{fontSize:'0.75rem'}}>demo@isro.gov.in / ecosight123</code></p>}
        </form>
      </div>
    </div>
  );
}

function ProtectedOverlay({ onLoginClick }) {
  return (
    <div className="protected-overlay">
      <div className="protected-content">
        <div className="protected-icon"><Lock size={32} color="var(--accent)"/></div>
        <h2>Secure Access Required</h2>
        <p>This module contains sensitive satellite and environmental data.</p>
        <button className="protected-login-btn" onClick={onLoginClick}>
          <Shield size={18}/> Sign In to Access <ChevronRight size={16}/>
        </button>
      </div>
    </div>
  );
}

// ── CHATBOT ────────────────────────────────────────────────────
function ChatPanel({ isOpen, onClose }) {
  const [input,setInput]       = useState('');
  const [messages,setMessages] = useState([{
    role:'system',
    content:'**GeoDrishti Omni-AI** — ML-powered predictions.\n\nI use trained ML models for:\n• Disaster risk\n• Crop yield predictions\n• Forest/deforestation alerts\n• Drought analysis\n• Urbanization forecasts\n\nPlus live weather & satellite data. Ask me anything!'
  }]);
  const [loading,setLoading] = useState(false);
  const bottomRef = useRef(null); const inputRef = useRef(null);
  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:'smooth'}); },[messages]);
  useEffect(()=>{ if(isOpen) inputRef.current?.focus(); },[isOpen]);

  const send = async(text) => {
    const q=(text||input).trim(); if(!q||loading) return;
    setMessages(p=>[...p,{role:'user',content:q}]); setInput(''); setLoading(true);
    try {
      const res=await fetch('http://127.0.0.1:8000/chat',{
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:q}) });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const d=await res.json();
      setMessages(p=>[...p,{role:'system',content:d.reply}]);
    } catch(err) {
      setMessages(p=>[...p,{role:'system',content:`⚠️ **Backend unreachable.**\nStart: \`uvicorn main:app --reload\`\n(${err.message})`}]);
    } finally { setLoading(false); }
  };

  const renderContent = (text) =>
    text.split(/(\*\*[^*]+\*\*)/g).map((p,i)=>
      p.startsWith('**')&&p.endsWith('**') ? <strong key={i}>{p.slice(2,-2)}</strong> : <span key={i}>{p}</span>
    );

  const SUGGESTIONS = ['Weather in Delhi today','Forest cover Maharashtra','Crop yield Punjab','Drought analysis','Disaster risk Chennai','NDVI for Kerala'];

  return (
    <section className={`chat-panel ${isOpen?'open':''} glass-card`} aria-label="AI assistant" aria-hidden={!isOpen}>
      <div className="chat-header">
        <div className="chat-title">
          <span className="chat-title-dot" aria-hidden="true"/>
          <MessageSquare size={16} aria-hidden="true"/>
          GeoDrishti Omni-AI
          <span style={{fontSize:'0.6rem',color:'var(--text-dim)',marginLeft:4,fontFamily:'var(--font-mono)'}}>REAL DATA</span>
        </div>
        <button className="close-chat-btn" onClick={onClose} aria-label="Close">×</button>
      </div>
      <div className="chat-messages" role="log" aria-live="polite">
        {messages.map((m,i)=>(
          <div key={i} className={`message-bubble ${m.role}`}>
            {m.content.split('\n').map((line,j,arr)=>(<span key={j}>{renderContent(line)}{j<arr.length-1&&<br/>}</span>))}
          </div>
        ))}
        {loading && <div className="message-bubble system" aria-busy="true"><span className="typing-dots"><span/><span/><span/></span></div>}
        <div ref={bottomRef}/>
      </div>
      {messages.length<=1 && (
        <div className="chat-suggestions">
          {SUGGESTIONS.map(s=><button key={s} className="suggestion-chip" onClick={()=>send(s)}>{s}</button>)}
        </div>
      )}
      <form className="chat-input-area" onSubmit={e=>{e.preventDefault();send();}}>
        <input ref={inputRef} type="text" placeholder="Ask anything about environment, weather, crops…"
               value={input} onChange={e=>setInput(e.target.value)} disabled={loading}/>
        <button type="submit" className="send-btn" disabled={!input.trim()||loading}><Send size={15}/></button>
      </form>
    </section>
  );
}

// ── VAPI OVERLAY ───────────────────────────────────────────────
function VapiCallOverlay({ isActive, onEnd }) {
  const [secs,setSecs] = useState(0);
  useEffect(()=>{
    if(!isActive){setSecs(0);return;}
    const id=setInterval(()=>setSecs(s=>s+1),1000); return ()=>clearInterval(id);
  },[isActive]);
  if(!isActive) return null;
  const fmt=s=>`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  return (
    <div className="vapi-overlay" role="dialog" aria-modal="true">
      <div className="vapi-card glass-card">
        <div className="vapi-wave" aria-hidden="true">
          {[...Array(5)].map((_,i)=><span key={i} style={{animationDelay:`${i*0.12}s`}}/>)}
        </div>
        <div className="vapi-avatar"><Mic size={28} color="white"/></div>
        <div className="vapi-label">GeoDrishti Voice Assistant</div>
        <div className="vapi-timer">{fmt(secs)}</div>
        <p className="vapi-hint">Ask about weather, crop yields, forest data, or disasters</p>
        <button className="vapi-end-btn" onClick={onEnd}><PhoneOff size={20}/> End Call</button>
      </div>
    </div>
  );
}

// ── ALERT TOAST ────────────────────────────────────────────────
function AlertToast({ alert, onDismiss }) {
  const colors = { yellow:'#f59e0b', orange:'#f97316', red:'#ef4444' };
  const icons  = { yellow:<AlertTriangle size={18}/>, orange:<AlertTriangle size={18}/>, red:<AlertCircle size={18}/> };
  const c = colors[alert.level] || '#10b981';
  useEffect(()=>{
    // Auto-dismiss after 8s for yellow, never auto for red/orange
    if(alert.level==='yellow') {
      const t=setTimeout(()=>onDismiss(alert.id),8000); return ()=>clearTimeout(t);
    }
  },[alert.id, alert.level, onDismiss]);
  return (
    <div className="alert-toast" style={{borderColor:c, background:`rgba(${alert.level==='red'?'239,68,68':'245,158,11'},0.08)`}}
         role="alert" aria-live="assertive">
      <div className="alert-toast-icon" style={{color:c}}>{icons[alert.level]||<Bell size={18}/>}</div>
      <div className="alert-toast-body">
        <div className="alert-toast-title" style={{color:c}}>
          {alert.level==='red'?'🔴 CRITICAL ALERT':alert.level==='orange'?'🟠 WARNING':'🟡 HEADS UP'}
        </div>
        <div className="alert-toast-aoi">{alert.aoi?.name} — {alert.aoi?.center?.lat?.toFixed(3)}, {alert.aoi?.center?.lng?.toFixed(3)}</div>
        <div className="alert-toast-summary">{alert.summary}</div>
      </div>
      <button className="alert-toast-close" onClick={()=>onDismiss(alert.id)} aria-label="Dismiss">×</button>
    </div>
  );
}

// ── EMERGENCY PANEL ────────────────────────────────────────────
const EMERGENCY_CONTACTS = [
  { label:'Disaster Response',    number:'1078',         icon:<Phone size={14}/>,         color:'#ef4444', desc:'NDMA 24×7' },
  { label:'Agri-Kisan Helpline',  number:'1800-180-1551',icon:<Phone size={14}/>,         color:'#f59e0b', desc:'Free for farmers' },
  { label:'IMD Weather Alerts',   number:'1800-180-1717',icon:<CloudRain size={13}/>,     color:'#8b5cf6', desc:'Cyclone/flood alerts' },
  { label:'Forest Fire Control',  number:'1926',         icon:<AlertTriangle size={13}/>, color:'#f97316', desc:'MoEFCC helpline' },
];

function EmergencyPanel() {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="emergency-panel-v2">
      <button className="emergency-toggle" onClick={()=>setExpanded(v=>!v)} aria-expanded={expanded}>
        <span className="emergency-pulse" aria-hidden="true"/>
        <PhoneCall size={13} aria-hidden="true"/>
        <span>Emergency Dispatch</span>
        <ChevronRight size={13} style={{marginLeft:'auto',transform:expanded?'rotate(90deg)':'none',transition:'0.2s'}}/>
      </button>
      {expanded && (
        <div className="emergency-contacts">
          {EMERGENCY_CONTACTS.map(c=>(
            <a key={c.number} href={`tel:${c.number.replace(/-/g,'')}`} className="emergency-contact-card"
               style={{borderLeftColor:c.color}} aria-label={`Call ${c.label}: ${c.number}`}>
              <div className="ec-icon" style={{color:c.color}}>{c.icon}</div>
              <div className="ec-info">
                <div className="ec-label">{c.label}</div>
                <div className="ec-desc">{c.desc}</div>
              </div>
              <div className="ec-number" style={{color:c.color}}>{c.number}</div>
            </a>
          ))}
          <a href="https://ndma.gov.in" target="_blank" rel="noopener noreferrer" className="emergency-link">
            <ExternalLink size={11}/> ndma.gov.in — National Disaster Management
          </a>
        </div>
      )}
    </div>
  );
}

// ── CONSTANTS ──────────────────────────────────────────────────
const MODULES = [
  { id:'forest',     title:'Forest Cover',        icon:<TreePine size={18}/>,  color:'#10b981' },
  { id:'population', title:'Population Dynamics', icon:<Users size={18}/>,     color:'#3b82f6' },
  { id:'weather',    title:'Weather Projections', icon:<CloudRain size={18}/>, color:'#8b5cf6' },
  { id:'crop',       title:'Crop Yields',         icon:<Wheat size={18}/>,     color:'#f59e0b' },
  { id:'drought',    title:'Drought Prediction',  icon:<Droplets size={18}/>,  color:'#ef4444' },
];

// ── MAIN APP ───────────────────────────────────────────────────
export default function App() {
  const [activeView,setActiveView]     = useState('aggregate');
  const [theme,setTheme]               = useState('dark-theme');
  const [isCallActive,setIsCallActive] = useState(false);
  const [isChatOpen,setIsChatOpen]     = useState(false);
  const [showNotif,setShowNotif]       = useState(false);
  const [alerts,setAlerts]             = useState([]);
  const [toasts,setToasts]             = useState([]);  // shown as popups
  const [currentUser,setCurrentUser]   = useState(getStoredUser);
  const [showAuthModal,setShowAuthModal] = useState(false);

  const handleAuthSuccess = u => { setCurrentUser(u); setShowAuthModal(false); };
  const handleLogout      = () => { authLogout(); setCurrentUser(null); setActiveView('aggregate'); };
  const toggleVoice = () => {
    if(isCallActive){vapi.stop();setIsCallActive(false);}
    else{vapi.start(import.meta.env.VITE_VAPI_ASSISTANT);setIsCallActive(true);}
  };

  const handleAlertTriggered = alertObj => {
    const a = {...alertObj, id:Date.now()};
    setAlerts(p=>[a,...p]);
    setToasts(p=>[a,...p.slice(0,2)]); // show max 3 toasts at once
  };

  const dismissToast  = id => setToasts(p=>p.filter(t=>t.id!==id));
  const toggleTheme   = () => setTheme(t=>t==='dark-theme'?'light-theme':'dark-theme');

  const userInitials = currentUser?.name ? currentUser.name.split(' ').map(p=>p[0]).join('').slice(0,2).toUpperCase():'?';
  const isProtected  = id => id!=='aggregate'&&id!=='aoi';
  const currentModule = MODULES.find(m=>m.id===activeView);
  const unreadCount   = alerts.filter(a=>a.level!=='green').length;

  const STATS = [
    {label:'Active Satellites',value:'7',    delta:'+2',   color:'var(--accent)',  bg:'var(--accent-dim)',  icon:<Satellite size={18}/>},
    {label:'Districts Covered',value:'742',  delta:'98.4%',color:'var(--accent2)', bg:'var(--accent2-dim)', icon:<Activity size={18}/>},
    {label:'Alerts Active',    value:String(unreadCount||0),delta:'live',color:'var(--warning)',bg:'var(--warning-dim)',icon:<Bell size={18}/>},
    {label:'Model Accuracy',   value:'94.2%',delta:'+0.3%',color:'var(--accent)',  bg:'var(--accent-dim)',  icon:<Shield size={18}/>},
  ];

  return (
    <div className={`app-layout ${theme}`}>
      <a href="#main-content" className="skip-link">Skip to main content</a>

      {showAuthModal && <AuthModal onClose={()=>setShowAuthModal(false)} onAuthSuccess={handleAuthSuccess}/>}
      <VapiCallOverlay isActive={isCallActive} onEnd={()=>{vapi.stop();setIsCallActive(false);}}/>

      {/* ── ALERT TOASTS ── */}
      <div className="toast-container" aria-live="assertive" aria-atomic="false">
        {toasts.map(t=><AlertToast key={t.id} alert={t} onDismiss={dismissToast}/>)}
      </div>

      {/* ── SIDEBAR ── */}
      <aside className="sidebar" aria-label="Navigation">
        <div className="brand">
          <div className="brand-inner">
            <div className="brand-logo" aria-hidden="true"><Satellite size={20} color="#000"/></div>
            <div className="brand-text"><h2>GeoDrishti</h2><span>ISRO EcoSight v4.0</span></div>
          </div>
        </div>

        <nav className="nav-menu" aria-label="Modules">
          <button className={`nav-item ${activeView==='aggregate'?'active':''}`} onClick={()=>setActiveView('aggregate')} aria-current={activeView==='aggregate'?'page':undefined}>
            <LayoutDashboard size={18} aria-hidden="true"/><span>Aggregate Overview</span><span className="nav-item-dot" aria-hidden="true"/>
          </button>
          <button className={`nav-item ${activeView==='aoi'?'active':''}`} onClick={()=>setActiveView('aoi')} aria-current={activeView==='aoi'?'page':undefined}>
            <MapPin size={18} aria-hidden="true"/><span>AOI Monitor</span><span className="nav-item-dot" aria-hidden="true"/>
          </button>
          <div className="nav-divider" role="separator">Analysis Modules</div>
          {MODULES.map(mod=>(
            <button key={mod.id} className={`nav-item ${activeView===mod.id?'active':''}`} onClick={()=>setActiveView(mod.id)} aria-current={activeView===mod.id?'page':undefined}>
              <span aria-hidden="true">{mod.icon}</span><span>{mod.title}</span>
              {!currentUser && <Lock size={11} style={{marginLeft:'auto',opacity:0.35}}/>}
              <span className="nav-item-dot" aria-hidden="true"/>
            </button>
          ))}
        </nav>

        {/* REDESIGNED EMERGENCY PANEL */}
        <EmergencyPanel/>

        {currentUser && (
          <div className="sidebar-user">
            <div className="sidebar-user-avatar" aria-hidden="true">{userInitials}</div>
            <div className="sidebar-user-info">
              <div className="sidebar-user-name">{currentUser.name}</div>
              <div className="sidebar-user-role">{currentUser.role}</div>
            </div>
            <button className="sidebar-logout-btn" onClick={handleLogout} title="Sign out"><LogOut size={14}/></button>
          </div>
        )}
      </aside>

      {/* ── MAIN ── */}
      <main className="main-content" id="main-content" tabIndex={-1}>
        <div className="canvas-background" aria-hidden="true">
          <Canvas>
            <ambientLight intensity={theme==='dark-theme'?0.2:0.8}/>
            <directionalLight position={[5,3,5]} intensity={2}/>
            <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={1.5}/>
            {theme==='dark-theme' && <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1}/>}
            <Sphere args={[2.2,64,64]} position={[2,0,-2]}>
              <meshStandardMaterial color={theme==='dark-theme'?'#0f172a':'#e2e8f0'} roughness={0.8}/>
            </Sphere>
            <Sphere args={[2.25,32,32]} position={[2,0,-2]}>
              <meshBasicMaterial color={theme==='dark-theme'?'#10b981':'#3b82f6'} wireframe transparent opacity={0.15}/>
            </Sphere>
          </Canvas>
        </div>

        <header className="top-bar">
          <div className="page-title-group">
            <span className="page-breadcrumb">ISRO EcoSight / {activeView==='aggregate'?'Overview':activeView==='aoi'?'AOI Monitor':currentModule?.title||activeView}</span>
            <div className="page-title">{activeView==='aggregate'?'Aggregate Overview':activeView==='aoi'?'AOI Monitor — Pin & Analyse':currentModule?.title}</div>
          </div>
          <div className="top-bar-actions">
            <div className="status-dot" role="status" aria-label="System online"/>
            <div className="divider-vertical" aria-hidden="true"/>
            <button className="icon-btn" style={{position:'relative'}} onClick={()=>setShowNotif(v=>!v)}
                    aria-label={`Notifications${unreadCount>0?` (${unreadCount})`:''}`} aria-expanded={showNotif}>
              <Bell size={18}/>
              {unreadCount>0 && <span className="topbar-badge" aria-hidden="true">{unreadCount}</span>}
            </button>
            <button className="icon-btn" onClick={toggleTheme} aria-label={theme==='dark-theme'?'Light mode':'Dark mode'}>
              {theme==='dark-theme'?<Sun size={18}/>:<Moon size={18}/>}
            </button>
            <div className="divider-vertical" aria-hidden="true"/>
            {currentUser ? (
              <button className="user-menu-btn" onClick={handleLogout} aria-label={`${currentUser.name} — sign out`}>
                <div className="user-avatar-sm" aria-hidden="true">{userInitials}</div>
                <span className="user-name-sm">{currentUser.name}</span>
                <LogOut size={13} style={{color:'var(--text-dim)',marginLeft:2}} aria-hidden="true"/>
              </button>
            ) : (
              <button className="auth-btn" onClick={()=>setShowAuthModal(true)} aria-label="Sign in">
                <UserCircle size={18} aria-hidden="true"/><span>Sign In</span>
              </button>
            )}
          </div>
        </header>

        <div className={`notif-drawer ${showNotif?'open':''}`} role="region" aria-label="Notifications" aria-hidden={!showNotif}>
          <NotificationsPanel alerts={alerts} onDismiss={id=>setAlerts(p=>p.filter(a=>a.id!==id))} onClearAll={()=>setAlerts([])}/>
        </div>

        <div className="dashboard-overlay">
          {activeView==='aggregate' && (
            <>
              <div className="stats-bar">
                {STATS.map((s,i)=>(
                  <div key={i} className="stat-chip">
                    <div className="stat-icon" style={{background:s.bg}}><span style={{color:s.color}}>{s.icon}</span></div>
                    <div className="stat-info">
                      <div className="stat-label">{s.label}</div>
                      <div className="stat-value">{s.value}<span className="stat-delta up">{s.delta}</span></div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="aggregate-grid">
                {MODULES.map(mod=>(
                  <div key={mod.id} className="glass-card grid-card">
                    <div className="card-header" style={{borderBottomColor:mod.color}}>
                      <div className="card-header-accent" style={{background:mod.color}} aria-hidden="true"/>
                      <span aria-hidden="true">{mod.icon}</span>{mod.title}
                      <span className="card-badge badge-sync">PowerBI</span>
                    </div>
                    <div className="iframe-placeholder">
                      <div className="placeholder-icon" style={{background:`${mod.color}18`}}><span style={{color:mod.color}}>{mod.icon}</span></div>
                      <div className="placeholder-label">Embed PowerBI Report<span>Replace with your &lt;iframe&gt;</span></div>
                    </div>
                  </div>
                ))}
                <div className="glass-card grid-card" aria-live="polite">
                  <div className="card-header" style={{borderBottomColor:'#6366f1'}}>
                    <div className="card-header-accent" style={{background:'#6366f1'}} aria-hidden="true"/>
                    <Activity size={18} color="#6366f1" aria-hidden="true"/>Live System Feed
                    <span className="card-badge badge-live">Live</span>
                  </div>
                  <div className="iframe-placeholder" style={{padding:'12px 18px',alignItems:'flex-start',justifyContent:'flex-start'}}>
                    <div style={{width:'100%'}}>
                      {[
                        {label:'Vapi Voice',     status:isCallActive?'ON AIR':'STANDBY', color:isCallActive?'#ef4444':'var(--accent)'},
                        {label:'FastAPI',         status:'RUNNING',  color:'var(--accent)'},
                        {label:'Open-Meteo',      status:'LIVE',     color:'var(--accent)'},
                        {label:'NASA POWER',      status:'LIVE',     color:'var(--accent)'},
                        {label:'ML Models',       status:'LOADED',   color:'var(--accent)'},
                        {label:'Auth',            status:currentUser?'ACTIVE':'IDLE', color:currentUser?'var(--accent)':'var(--text-dim)'},
                        {label:'Alerts',          status:`${unreadCount} ACTIVE`, color:unreadCount>0?'var(--danger)':'var(--text-dim)'},
                      ].map((item,i)=>(
                        <div key={i} className="feed-entry">
                          <span className="feed-dot" style={{background:item.color}} aria-hidden="true"/>
                          <span style={{color:'var(--text-muted)',flex:1,fontSize:'0.78rem'}}>{item.label}</span>
                          <span style={{color:item.color,fontSize:'0.67rem',letterSpacing:'0.08em'}}>{item.status}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeView==='aoi' && (
            <div className="glass-card single-view-card" style={{overflow:'hidden'}}>
              <div className="card-header" style={{borderBottomColor:'var(--accent)'}}>
                <div className="card-header-accent" style={{background:'var(--accent)'}} aria-hidden="true"/>
                <MapPin size={18} style={{color:'var(--accent)'}} aria-hidden="true"/>
                AOI Monitor — Pin, Analyse &amp; Alert
                <span className="card-badge badge-live">Satellite</span>
              </div>
              <AOIPanel currentUser={currentUser} onAlertTriggered={handleAlertTriggered} theme={theme}/>
            </div>
          )}

          {activeView!=='aggregate'&&activeView!=='aoi' && (
            <div className="glass-card single-view-card" style={{position:'relative'}}>
              <div className="card-header" style={{borderBottomColor:currentModule?.color}}>
                <div className="card-header-accent" style={{background:currentModule?.color}} aria-hidden="true"/>
                <span aria-hidden="true">{currentModule?.icon}</span>
                {currentModule?.title} — Detailed View
                <span className="card-badge badge-sync">PowerBI</span>
              </div>
              <div className="iframe-placeholder full">
                <div className="placeholder-icon"><span style={{color:currentModule?.color}}>{currentModule?.icon}</span></div>
                <div className="placeholder-label">{currentModule?.title} Visualization<span>Paste your PowerBI &lt;iframe&gt; here</span></div>
              </div>
              {!currentUser&&isProtected(activeView) && <ProtectedOverlay onLoginClick={()=>setShowAuthModal(true)}/>}
            </div>
          )}
        </div>
      </main>

      <ChatPanel isOpen={isChatOpen} onClose={()=>setIsChatOpen(false)}/>

      <div className="fab-container" role="group" aria-label="Quick actions">
        <button className="fab chat-fab" aria-label={isChatOpen?'Close chat':'Open chat'} aria-expanded={isChatOpen} onClick={()=>setIsChatOpen(v=>!v)}>
          {isChatOpen?<X size={22} aria-hidden="true"/>:<MessageSquare size={22} aria-hidden="true"/>}
        </button>
        <button className={`fab ${isCallActive?'mic-active':'mic-inactive'}`}
                aria-label={isCallActive?'End call':'Start voice assistant'} aria-pressed={isCallActive} onClick={toggleVoice}>
          {isCallActive?<PhoneOff size={22} aria-hidden="true"/>:<Mic size={22} aria-hidden="true"/>}
        </button>
      </div>
    </div>
  );
}