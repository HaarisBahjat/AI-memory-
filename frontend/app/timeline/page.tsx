"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { episodesApi, type Episode } from "@/lib/api";
import { Line, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, BarElement, Title, Tooltip, Legend, Filler
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

export default function TimelinePage() {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string|null>(null);
  const [expanded, setExpanded] = useState<string|null>(null);

  useEffect(()=>{
    episodesApi.list()
      .then((r: any)=>{ setEpisodes(r.data.items??[]); setLoading(false); })
      .catch((err: any)=>{
        const status = err?.response?.status;
        console.warn("Timeline fetch failed:", err);
        if (status === 401) {
          window.location.href = "/login";
        } else {
          setError(`Failed to load timeline (HTTP ${status ?? "network error"})`);
        }
        setLoading(false);
      });
  },[]);

  const sorted = [...episodes].sort((a,b)=>new Date(a.timestamp).getTime()-new Date(b.timestamp).getTime());
  const last30  = sorted.slice(-30);
  const labels  = last30.map(e=>new Date(e.timestamp).toLocaleDateString("en-US",{month:"short",day:"numeric"}));
  const moods   = last30.map(e=>e.extracted_metrics?.moodScore??null);

  const stressorCounts: Record<string,number> = {};
  episodes.forEach(e=>{ const s=e.extracted_metrics?.primaryStressor; if(s&&s!=="N/A") stressorCounts[s]=(stressorCounts[s]??0)+1; });
  const topStressors = Object.entries(stressorCounts).sort((a,b)=>b[1]-a[1]).slice(0,6);

  const chartOptions = {
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{backgroundColor:"#1C1C28",borderColor:"rgba(255,255,255,.1)",borderWidth:1,titleColor:"#F1F1F5",bodyColor:"#A0A0BA"}},
    scales:{x:{grid:{color:"rgba(255,255,255,.05)"},ticks:{color:"#5C5C7A",font:{size:11}}},y:{grid:{color:"rgba(255,255,255,.05)"},ticks:{color:"#5C5C7A",font:{size:11}}}}
  };

  const moodData = {
    labels,
    datasets:[{label:"Mood",data:moods,borderColor:"#7C6FF7",backgroundColor:"rgba(124,111,247,.12)",tension:.4,fill:true,pointBackgroundColor:"#7C6FF7",pointRadius:4}]
  };
  const stressorData = {
    labels:topStressors.map(([k])=>k),
    datasets:[{label:"Frequency",data:topStressors.map(([,v])=>v),backgroundColor:"rgba(124,111,247,.6)",borderColor:"#7C6FF7",borderWidth:1,borderRadius:6}]
  };

  return (
    <div style={{display:"flex",minHeight:"100vh"}}>
      <Sidebar />
      <main style={{marginLeft:"220px",flex:1,padding:"32px",maxWidth:"1100px"}}>
        <div style={{marginBottom:"28px"}}>
          <h1 style={{fontSize:"28px",fontWeight:800}}>Health Timeline</h1>
          <p className="text-muted" style={{fontSize:"14px",marginTop:"6px"}}>{episodes.length} sessions recorded</p>
        </div>

        {error && (
          <div style={{background:"var(--danger-subtle,#3a1a1a)",border:"1px solid var(--danger,#f66)",borderRadius:"10px",padding:"16px 20px",marginBottom:"20px",color:"var(--danger,#f66)"}}>
            ⚠️ {error} — <a href="/login" style={{color:"inherit",textDecoration:"underline"}}>Re-login</a> if your session expired.
          </div>
        )}

        {loading ? (
          <div style={{display:"flex",justifyContent:"center",padding:"60px"}}><span className="spinner" style={{width:"32px",height:"32px"}} /></div>
        ) : (
          <>
            {/* Charts */}
            <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:"20px",marginBottom:"32px"}}>
              <div className="card">
                <h3 style={{fontSize:"14px",fontWeight:600,marginBottom:"20px",color:"var(--text-secondary)"}}>Mood Score (Last 30 Days)</h3>
                <div style={{height:"200px"}}><Line data={moodData} options={chartOptions as any} /></div>
              </div>
              <div className="card">
                <h3 style={{fontSize:"14px",fontWeight:600,marginBottom:"20px",color:"var(--text-secondary)"}}>Top Stressors</h3>
                <div style={{height:"200px"}}><Bar data={stressorData} options={chartOptions as any} /></div>
              </div>
            </div>

            {/* Episode cards */}
            <div style={{display:"flex",flexDirection:"column",gap:"16px"}}>
              {[...episodes].reverse().map((ep,i)=>{
                const m=ep.extracted_metrics;
                const mood=m?.moodScore;
                const moodColor=mood==null?"var(--text-muted)":mood>=7?"var(--success)":mood>=5?"var(--warning)":"var(--danger)";
                const isOpen=expanded===ep.id;
                return (
                  <div key={ep.id} className="card animate-fadeInUp" style={{animationDelay:`${i*30}ms`,cursor:"pointer"}}
                    onClick={()=>setExpanded(isOpen?null:ep.id)}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                      <div style={{display:"flex",gap:"16px",alignItems:"center"}}>
                        <div style={{width:"48px",height:"48px",borderRadius:"12px",background:"var(--accent-subtle)",border:"1px solid var(--border-accent)",display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column"}}>
                          <span style={{fontSize:"16px",fontWeight:800,color:moodColor}}>{mood??"-"}</span>
                          <span style={{fontSize:"9px",color:"var(--text-muted)"}}>mood</span>
                        </div>
                        <div>
                          <p style={{fontWeight:600,fontSize:"14px"}}>{new Date(ep.timestamp).toLocaleDateString("en-US",{weekday:"long",month:"long",day:"numeric"})}</p>
                          <p className="text-muted" style={{fontSize:"13px"}}>😴 {m?.sleepHoursLogged??"-"}h sleep · ⚡ {m?.primaryStressor??"Unknown stressor"}</p>
                        </div>
                      </div>
                      <span style={{color:"var(--text-muted)",transition:"transform .2s",display:"inline-block",transform:isOpen?"rotate(90deg)":"none"}}>›</span>
                    </div>
                    {isOpen && (
                      <div style={{marginTop:"16px",paddingTop:"16px",borderTop:"1px solid var(--border)"}}>
                        <p style={{fontSize:"14px",lineHeight:1.8,color:"var(--text-secondary)"}}>{ep.session_summary}</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
