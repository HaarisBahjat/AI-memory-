"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import styles from "./login.module.css";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      const { data } = await authApi.login(email, password);
      login(data.access_token, data.refresh_token);
      router.push("/chat");
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const msg = Array.isArray(detail) ? detail[0].msg : (typeof detail === 'string' ? detail : "Login failed. Check credentials.");
      setError(msg);
    } finally { setLoading(false); }
  };

  return (
    <div className={styles.page}>
      <div className={styles.visual}>
        <div className={styles.orb1} />
        <div className={styles.orb2} />
        <div className={styles.visualContent}>
          <h1 className="gradient-text" style={{fontSize:"42px",fontWeight:800,lineHeight:1.1}}>
            Your Mind,<br/>Remembered.
          </h1>
          <p style={{color:"var(--text-secondary)",marginTop:"16px",fontSize:"16px",maxWidth:"320px",lineHeight:1.7}}>
            An AI wellness companion that builds a deep, persistent understanding of you over time.
          </p>
          <div className={styles.stats}>
            {[["3 Layers","Memory architecture"],["7.5 Phases","Built & shipped"],["< 100ms","Graph retrieval"]].map(([v,l])=>(
              <div key={v} className={styles.stat}><span className={styles.statVal}>{v}</span><span className={styles.statLbl}>{l}</span></div>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.formSide}>
        <div className={styles.formCard + " animate-fadeInUp"}>
          <h2 style={{fontSize:"24px",fontWeight:700,marginBottom:"8px"}}>Welcome back</h2>
          <p className="text-muted" style={{fontSize:"14px",marginBottom:"32px"}}>Sign in to continue your journey</p>

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.field}>
              <label>Email</label>
              <input className={"input" + (error ? " error" : "")} type="email" placeholder="you@example.com"
                value={email} onChange={e=>setEmail(e.target.value)} required />
            </div>
            <div className={styles.field}>
              <label>Password</label>
              <input className={"input" + (error ? " error" : "")} type="password" placeholder="••••••••"
                value={password} onChange={e=>setPassword(e.target.value)} required />
            </div>

            {error && <div className={styles.errorBox}>{error}</div>}

            <button type="submit" className="btn btn-primary btn-lg" style={{width:"100%"}} disabled={loading}>
              {loading ? <><span className="spinner" style={{width:"16px",height:"16px"}} /> Signing in…</> : "Sign In →"}
            </button>
          </form>

          <p style={{textAlign:"center",marginTop:"24px",fontSize:"14px",color:"var(--text-muted)"}}>
            No account? <Link href="/register">Create one</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
