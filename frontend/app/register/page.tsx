"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { useAuthStore } from "@/store/auth";

export default function RegisterPage() {
  const router = useRouter();
  const { login } = useAuthStore();
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);

  const strength = password.length === 0 ? 0 : password.length < 6 ? 1 : password.length < 10 ? 2 : /[A-Z]/.test(password) && /\d/.test(password) ? 4 : 3;
  const strengthLabel = ["","Weak","Fair","Good","Strong"][strength];
  const strengthColor = ["","var(--danger)","var(--warning)","var(--accent)","var(--success)"][strength];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(""); setLoading(true);
    try {
      const { data } = await authApi.register(email, password);
      login(data.access_token, data.refresh_token);
      router.push("/chat");
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const msg = Array.isArray(detail) ? detail[0].msg : (typeof detail === 'string' ? detail : "Registration failed.");
      setError(msg);
    } finally { setLoading(false); }
  };

  return (
    <div style={{minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",padding:"24px",background:"var(--bg-base)"}}>
      <div className="card animate-fadeInUp" style={{width:"100%",maxWidth:"440px",padding:"40px"}}>
        <h1 style={{fontSize:"28px",fontWeight:800,marginBottom:"8px"}}>Create Account</h1>
        <p className="text-muted" style={{fontSize:"14px",marginBottom:"32px"}}>Start building your AI memory today</p>

        <form onSubmit={handleSubmit} style={{display:"flex",flexDirection:"column",gap:"20px"}}>
          <div style={{display:"flex",flexDirection:"column",gap:"8px"}}>
            <label style={{fontSize:"13px",fontWeight:600,color:"var(--text-secondary)"}}>Email</label>
            <input className="input" type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required />
          </div>
          <div style={{display:"flex",flexDirection:"column",gap:"8px"}}>
            <label style={{fontSize:"13px",fontWeight:600,color:"var(--text-secondary)"}}>Password</label>
            <input className="input" type="password" placeholder="Min. 8 characters" value={password} onChange={e=>setPassword(e.target.value)} required minLength={8} />
            {password && (
              <div style={{display:"flex",alignItems:"center",gap:"10px",marginTop:"4px"}}>
                <div style={{flex:1,height:"4px",borderRadius:"2px",background:"var(--border)"}}>
                  <div style={{height:"100%",borderRadius:"2px",width:`${strength*25}%`,background:strengthColor,transition:"all .3s ease"}} />
                </div>
                <span style={{fontSize:"11px",color:strengthColor,fontWeight:600,minWidth:"40px"}}>{strengthLabel}</span>
              </div>
            )}
          </div>

          {error && <div style={{background:"var(--danger-subtle)",border:"1px solid rgba(248,113,113,.25)",borderRadius:"var(--radius-md)",padding:"12px 16px",fontSize:"13px",color:"var(--danger)"}}>{error}</div>}

          <button type="submit" className="btn btn-primary btn-lg" style={{width:"100%"}} disabled={loading}>
            {loading ? <><span className="spinner" style={{width:"16px",height:"16px"}} /> Creating account…</> : "Create Account →"}
          </button>
        </form>

        <p style={{textAlign:"center",marginTop:"24px",fontSize:"14px",color:"var(--text-muted)"}}>
          Already have an account? <Link href="/login">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
