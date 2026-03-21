import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Stars } from '@react-three/drei';
import { LayoutDashboard, TreePine, Users, CloudRain, Wheat, Droplets, Mic, MessageSquare, UserCircle, PhoneCall, Sun, Moon, Bell, Download, Settings, Activity } from 'lucide-react';
import VapiModule from '@vapi-ai/web'; 
import './App.css';

// --- VITE IMPORT FIX ---
const Vapi = VapiModule.default || VapiModule;
const vapi = new Vapi(import.meta.env.VITE_VAPI_API_KEY); 

console.log("VAPI Initialized with API Key:", import.meta.env.VITE_VAPI_API_KEY);

export default function App() {
  const [activeView, setActiveView] = useState('aggregate');
  const [theme, setTheme] = useState('dark-theme');

  // --- VAPI VOICE ASSISTANT SETUP ---
  const [isCallActive, setIsCallActive] = useState(false);

  const toggleVoiceAssistant = () => {
    if (isCallActive) {
      vapi.stop();
      setIsCallActive(false);
    } else {
      // Calling the dashboard bot directly. No overrides.
      vapi.start(import.meta.env.VITE_VAPI_ASSISTANT); 
      setIsCallActive(true);
    }
  };

  // --- TEXT CHATBOT STATE ---
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', content: 'GeoDristri Omni-AI initialized. How can I assist you with environmental data today?' }
  ]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    // 1. Immediately show user's message
    const userMsg = chatInput;
    const newMessages = [...messages, { role: 'user', content: userMsg }];
    setMessages(newMessages);
    setChatInput('');

    try {
      // 2. Ping your real FastAPI backend
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg })
      });
      
      const data = await response.json();
      
      // 3. Display the actual backend response
      setMessages([...newMessages, { role: 'system', content: data.reply }]);
      
    } catch (error) {
      setMessages([...newMessages, { role: 'system', content: "⚠️ API Offline. Please make sure your Python uvicorn server is running on port 8000." }]);
    }
  };

  const toggleTheme = () => {
    setTheme(theme === 'dark-theme' ? 'light-theme' : 'dark-theme');
  };

  const modules = [
    { id: 'forest', title: 'Forest Cover', icon: <TreePine size={20}/>, color: '#10b981' },
    { id: 'population', title: 'Population Dynamics', icon: <Users size={20}/>, color: '#3b82f6' },
    { id: 'weather', title: 'Weather Projections', icon: <CloudRain size={20}/>, color: '#8b5cf6' },
    { id: 'crop', title: 'Crop Yields', icon: <Wheat size={20}/>, color: '#f59e0b' },
    { id: 'drought', title: 'Drought Prediction', icon: <Droplets size={20}/>, color: '#ef4444' }
  ];

  return (
    <div className={`app-layout ${theme}`}>
      
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <h2>🚀 GeoDrishti</h2>
        </div>
        
        <nav className="nav-menu">
          <button 
            className={`nav-item ${activeView === 'aggregate' ? 'active' : ''}`}
            onClick={() => setActiveView('aggregate')}
          >
            <LayoutDashboard size={20} />
            <span>Aggregate Overview</span>
          </button>
          
          <div className="nav-divider">Analysis Modules</div>
          
          {modules.map((mod) => (
            <button 
              key={mod.id}
              className={`nav-item ${activeView === mod.id ? 'active' : ''}`}
              onClick={() => setActiveView(mod.id)}
            >
              {mod.icon}
              <span>{mod.title}</span>
            </button>
          ))}
        </nav>

        {/* EMERGENCY PANEL */}
        <div className="emergency-panel">
          <div className="emergency-header">
            <PhoneCall size={16} /> Emergency Dispatch
          </div>
          <ul className="emergency-list">
            <li>Disaster Response: <strong>1078</strong></li>
            <li>Agri-Kisan: <strong>1800-180-1551</strong></li>
          </ul>
        </div>
      </aside>

      {/* MAIN WORKSPACE */}
      <main className="main-content">
        
        {/* 3D HOLOGRAPHIC GLOBE ENGINE */}
        <div className="canvas-background">
          <Canvas>
            <ambientLight intensity={theme === 'dark-theme' ? 0.2 : 0.8} />
            <directionalLight position={[5, 3, 5]} intensity={2} />
            <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={1.5} />
            
            {theme === 'dark-theme' && (
              <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
            )}

            <Sphere args={[2.2, 64, 64]} position={[2, 0, -2]}>
              <meshStandardMaterial 
                color={theme === 'dark-theme' ? "#0f172a" : "#e2e8f0"} 
                roughness={0.8}
              />
            </Sphere>
            
            <Sphere args={[2.25, 32, 32]} position={[2, 0, -2]}>
              <meshBasicMaterial 
                color={theme === 'dark-theme' ? "#10b981" : "#3b82f6"} 
                wireframe={true}
                transparent={true}
                opacity={0.15}
              />
            </Sphere>
          </Canvas>
        </div>

        {/* TOP BAR UI */}
        <header className="top-bar">
          <div className="page-title">
            {activeView === 'aggregate' ? 'Aggregate Overview' : modules.find(m => m.id === activeView)?.title}
          </div>
          
          <div className="top-bar-actions">
            {/* INJECTED JARVIS VOICE ASSISTANT BUTTON */}

            <div className="divider-vertical"></div>

            <button className="icon-btn theme-toggle" onClick={toggleTheme} title="Toggle Theme">
              {theme === 'dark-theme' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <button className="auth-btn">
              <UserCircle size={20} />
              <span>Login</span>
            </button>
          </div>
        </header>

        {/* DYNAMIC DASHBOARD OVERLAY */}
        <div className="dashboard-overlay">
          
          {activeView === 'aggregate' && (
            <div className="aggregate-grid">
              {modules.map(mod => (
                <div key={mod.id} className="glass-card grid-card">
                  <div className="card-header" style={{ borderBottomColor: mod.color }}>
                    {mod.icon} {mod.title}
                  </div>
                  <div className="iframe-placeholder">
                    <p>Syncing datasets...</p>
                  </div>
                </div>
              ))}
              
              {/* The 6th Balancing Card */}
              <div className="glass-card grid-card">
                <div className="card-header" style={{ borderBottomColor: '#6366f1' }}>
                  <Activity size={20} color="#6366f1" /> Live System Feed
                </div>
                <div className="iframe-placeholder">
                  <div style={{ textAlign: 'left', width: '100%' }}>
                    <p style={{ color: '#10b981', margin: '5px 0' }}>● Vapi Voice Server Active</p>
                    <p style={{ color: '#10b981', margin: '5px 0' }}>● FastAPI Models Loaded</p>
                    <p style={{ color: '#f59e0b', margin: '5px 0' }}>● Syncing Synthetic Datasets...</p>
                  </div>
                </div>
              </div>

            </div>
          )}

          {activeView !== 'aggregate' && (
            <div className="glass-card single-view-card">
              <div className="iframe-placeholder full">
                <h2>{modules.find(m => m.id === activeView)?.title} Visualization</h2>
                <p>Waiting for iframe injection...</p>
              </div>
            </div>
          )}

        </div>
      </main>

      {/* --- SLIDING CHATBOT PANEL --- */}
      <div className={`chat-panel ${isChatOpen ? 'open' : ''} glass-card`}>
        <div className="chat-header">
          <div className="chat-title">
            <MessageSquare size={18} /> GeoDrishti Omni-AI
          </div>
          <button className="close-chat-btn" onClick={() => setIsChatOpen(false)}>×</button>
        </div>
        
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message-bubble ${msg.role}`}>
              {msg.content}
            </div>
          ))}
        </div>

        <form className="chat-input-area" onSubmit={handleSendMessage}>
          <input 
            type="text" 
            placeholder="Ask about forests, weather, crops..." 
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
          />
          <button type="submit" className="send-btn">Send</button>
        </form>
      </div>

      {/* FLOATING ACTION BUTTONS */}
      <div className="fab-container">
        <button 
          className="fab chat-fab" 
          title="Open Text Chatbot"
          onClick={() => setIsChatOpen(!isChatOpen)}
        >
          <MessageSquare size={24} />
        </button>
        <button 
          className={isCallActive ? "fab mic-active" : "fab voice-fab"} 
          title={isCallActive ? "End Voice Call" : "Initialize Voice Assistant"}
          onClick={toggleVoiceAssistant}
          style={{ 
            backgroundColor: isCallActive ? '#ef4444' : '#3b82f6',
            boxShadow: isCallActive ? '0 0 20px rgba(239, 68, 68, 0.8)' : ''
          }}
        >
          <Mic size={24} color="white" />
        </button>
      </div>

    </div>
  );
}