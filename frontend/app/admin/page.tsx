"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { adminApi } from "@/lib/api";

interface TriageEvent { id:string; crisis_type:string; severity:string; created_at:string; alert_sent:boolean; session_hash?:string; }

export default function AdminPage() {
  const [triage, setTriage]      = useState<TriageEvent[]>([]);
  const [loading, setLoading]    = useState(true);
  const [consolStatus, setConsolStatus] = useState<string|null>(null);
  const [polling, setPolling]    = useState(false);

  useEffect(()=>{
    adminApi.triageEvents().then((r: any)=>{ setTriage(r.data.events??[]); setLoading(false); }).catch(()=>setLoading(false));
  },[]);

  const triggerConsolidate = async () => {
    setConsolStatus("Triggered…");
    try {
      await adminApi.triggerConsolidate();
      setConsolStatus("Running…");
      setPolling(true);
      let attempts = 0;
      const interval = setInterval(async () => {
        try {
          const { data } = await adminApi.consolidateStatus();
          setConsolStatus(`Status: ${data.status} (${data.processed??0} processed)`);
          if (data.status === "COMPLETED" || data.status === "IDLE" || attempts > 10) {
            clearInterval(interval); setPolling(false);
          }
        } catch {}
        attempts++;
      }, 3000);
    } catch (e: any) {
      setConsolStatus("Failed: " + (e.response?.data?.detail ?? "Unknown error"));
    }
  };

  const sevColor = (s:string) => s==="HIGH"?"danger":s==="MEDIUM"?"warning":"muted";

  return (
    <div style={{display:"flex",minHeight:"100vh"}}>
      <Sidebar />
      <main style={{marginLeft:"220px",flex:1,padding:"32px",maxWidth:"1100px"}}>
        <div style={{marginBottom:"28px"}}>
          <h1 style={{fontSize:"28px",fontWeight:800}}>Admin Panel</h1>
          <p className="text-muted" style={{fontSize:"14px",marginTop:"6px"}}>System controls and triage event monitor</p>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"20px",marginBottom:"32px"}}>
          {/* Consolidation Card */}
          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:"8px"}}>⚙️ Batch Consolidation</h3>
            <p className="text-muted" style={{fontSize:"13px",marginBottom:"20px",lineHeight:1.6}}>
              Runs the nightly batch job immediately. Extracts semantic memories and graph triples from PENDING episodes.
            </p>
            {consolStatus && (
              <div style={{background:"var(--accent-subtle)",border:"1px solid var(--border-accent)",borderRadius:"var(--radius-md)",padding:"10px 14px",fontSize:"13px",color:"var(--accent)",marginBottom:"16px",display:"flex",alignItems:"center",gap:"8px"}}>
                {polling && <span className="spinner" style={{width:"14px",height:"14px"}} />}
                {consolStatus}
              </div>
            )}
            <button className="btn btn-primary" onClick={triggerConsolidate} disabled={polling}>
              {polling ? "Running…" : "Run Consolidation Now"}
            </button>
          </div>

          {/* Stats Card */}
          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:"16px"}}>📊 Quick Stats</h3>
            <div style={{display:"flex",flexDirection:"column",gap:"12px"}}>
              {[["Total Triage Events", triage.length],["High Severity", triage.filter(t=>t.severity==="HIGH").length],["Alerts Sent", triage.filter(t=>t.alert_sent).length]].map(([l,v])=>(
                <div key={l as string} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"10px 0",borderBottom:"1px solid var(--border)"}}>
                  <span style={{fontSize:"13px",color:"var(--text-secondary)"}}>{l}</span>
                  <span style={{fontWeight:700,fontSize:"16px"}}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Triage Table */}
        <div className="card" style={{padding:0,overflow:"hidden"}}>
          <div style={{padding:"20px 24px",borderBottom:"1px solid var(--border)"}}>
            <h3 style={{fontWeight:700}}>🚨 Triage Events</h3>
          </div>
          {loading ? (
            <div style={{display:"flex",justifyContent:"center",padding:"40px"}}><span className="spinner" style={{width:"28px",height:"28px"}} /></div>
          ) : triage.length===0 ? (
            <div style={{padding:"40px",textAlign:"center",color:"var(--text-muted)"}}>No triage events recorded.</div>
          ) : (
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse"}}>
                <thead>
                  <tr style={{background:"var(--bg-surface)"}}>
                    {["Date","Crisis Type","Severity","Alert Sent","Session Hash"].map(h=>(
                      <th key={h} style={{padding:"12px 20px",textAlign:"left",fontSize:"12px",fontWeight:600,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:".04em",borderBottom:"1px solid var(--border)"}}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {triage.map((ev,i)=>(
                    <tr key={ev.id} style={{background:i%2===0?"transparent":"rgba(255,255,255,.02)",transition:"background var(--ease-fast)"}}>
                      <td style={{padding:"14px 20px",fontSize:"13px"}}>{new Date(ev.created_at).toLocaleDateString()}</td>
                      <td style={{padding:"14px 20px",fontSize:"13px",fontWeight:500}}>{ev.crisis_type}</td>
                      <td style={{padding:"14px 20px"}}><span className={`badge badge-${sevColor(ev.severity)}`}>{ev.severity}</span></td>
                      <td style={{padding:"14px 20px"}}><span className={`dot-${ev.alert_sent?"online":"offline"}`} /></td>
                      <td style={{padding:"14px 20px",fontSize:"12px",color:"var(--text-muted)",fontFamily:"monospace"}}>{ev.session_hash?.slice(0,16)??"-"}…</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
