"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { memoriesApi, type Memory } from "@/lib/api";

const CATEGORIES = ["ALL","TRIGGER","COPING_MECHANISM","SYMPTOM","BASELINE","GOAL","ACTIVITY","PERSON","EVENT"];
const CATEGORY_COLOR: Record<string,string> = {
  TRIGGER:"danger", COPING_MECHANISM:"success", SYMPTOM:"warning",
  BASELINE:"accent", GOAL:"accent", ACTIVITY:"success", PERSON:"muted", EVENT:"muted"
};

export default function MemoriesPage() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [filter, setFilter]     = useState("ALL");
  const [loading, setLoading]   = useState(true);
  const [search, setSearch]     = useState("");

  useEffect(() => {
    memoriesApi.list().then((r: any) => { setMemories(r.data.memories ?? []); setLoading(false); }).catch(()=>setLoading(false));
  }, []);

  const toggle = async (m: Memory) => {
    await memoriesApi.pin(m.id, !m.is_pinned);
    setMemories(prev => prev.map(x => x.id===m.id ? {...x, is_pinned:!x.is_pinned} : x));
  };
  const del = async (id: string) => {
    await memoriesApi.delete(id);
    setMemories(prev => prev.filter(x => x.id!==id));
  };

  const visible = memories
    .filter(m => filter==="ALL" || m.category===filter)
    .filter(m => !search || m.text.toLowerCase().includes(search.toLowerCase()))
    .sort((a,b) => (b.is_pinned?1:0)-(a.is_pinned?1:0));

  const pinned  = visible.filter(m=>m.is_pinned);
  const regular = visible.filter(m=>!m.is_pinned);

  return (
    <div style={{display:"flex",minHeight:"100vh"}}>
      <Sidebar />
      <main style={{marginLeft:"220px",flex:1,padding:"32px",maxWidth:"1100px"}}>
        <div style={{marginBottom:"28px"}}>
          <h1 style={{fontSize:"28px",fontWeight:800}}>Long-Term Memory</h1>
          <p className="text-muted" style={{fontSize:"14px",marginTop:"6px"}}>{memories.length} facts extracted from your sessions</p>
        </div>

        {/* Filters */}
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap",marginBottom:"20px"}}>
          {CATEGORIES.map(c => (
            <button key={c} onClick={()=>setFilter(c)}
              className={`btn btn-sm ${filter===c?"btn-primary":"btn-ghost"}`}
              style={{textTransform:"uppercase",letterSpacing:".04em",fontSize:"11px"}}>
              {c.replace("_"," ")}
            </button>
          ))}
        </div>

        <input className="input" placeholder="🔍 Search memories…" value={search}
          onChange={e=>setSearch(e.target.value)} style={{marginBottom:"24px",maxWidth:"400px"}} />

        {loading ? (
          <div style={{display:"flex",justifyContent:"center",padding:"60px"}}><span className="spinner" style={{width:"32px",height:"32px"}} /></div>
        ) : memories.length===0 ? (
          <div className="card" style={{textAlign:"center",padding:"60px",color:"var(--text-muted)"}}>
            <div style={{fontSize:"48px",marginBottom:"16px"}}>🧠</div>
            <p style={{fontSize:"16px",fontWeight:600}}>No memories yet</p>
            <p style={{fontSize:"14px",marginTop:"8px"}}>Chat with the AI for a few sessions, then run consolidation to extract memories.</p>
          </div>
        ) : (
          <>
            {pinned.length > 0 && (
              <>
                <p style={{fontSize:"12px",fontWeight:700,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:".06em",marginBottom:"12px"}}>📌 Pinned</p>
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(300px,1fr))",gap:"16px",marginBottom:"28px"}}>
                  {pinned.map(m=><MemoryCard key={m.id} m={m} onPin={toggle} onDelete={del} />)}
                </div>
              </>
            )}
            {regular.length > 0 && (
              <>
                {pinned.length>0 && <p style={{fontSize:"12px",fontWeight:700,color:"var(--text-muted)",textTransform:"uppercase",letterSpacing:".06em",marginBottom:"12px"}}>All Facts</p>}
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(300px,1fr))",gap:"16px"}}>
                  {regular.map(m=><MemoryCard key={m.id} m={m} onPin={toggle} onDelete={del} />)}
                </div>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function MemoryCard({ m, onPin, onDelete }: { m: Memory; onPin: (m:Memory)=>void; onDelete:(id:string)=>void }) {
  const badgeCls = `badge badge-${CATEGORY_COLOR[m.category]??"muted"}`;
  return (
    <div className="card animate-fadeInUp" style={{position:"relative",borderColor:m.is_pinned?"var(--border-accent)":"var(--border)"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:"12px"}}>
        <span className={badgeCls}>{m.category.replace("_"," ")}</span>
        <div style={{display:"flex",gap:"6px"}}>
          <button onClick={()=>onPin(m)} className="btn btn-ghost btn-sm" data-tooltip={m.is_pinned?"Unpin":"Pin"}>
            {m.is_pinned ? "📌" : "🔗"}
          </button>
          <button onClick={()=>onDelete(m.id)} className="btn btn-danger btn-sm" data-tooltip="Delete">✕</button>
        </div>
      </div>
      <p style={{fontSize:"14px",lineHeight:1.7,color:"var(--text-primary)",marginBottom:"14px"}}>{m.text}</p>
      <div style={{display:"flex",alignItems:"center",gap:"12px"}}>
        <div style={{flex:1,height:"3px",borderRadius:"2px",background:"var(--border)"}}>
          <div style={{height:"100%",borderRadius:"2px",width:`${Math.min(m.reinforcement_count*20,100)}%`,background:"var(--accent)"}} />
        </div>
        <span style={{fontSize:"11px",color:"var(--text-muted)",whiteSpace:"nowrap"}}>Confirmed {m.reinforcement_count}×</span>
      </div>
    </div>
  );
}
