"use client";
import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import { chatApi } from "@/lib/api";
import styles from "./chat.module.css";

interface Message { role: "user"|"assistant"; content: string; timestamp: Date; graphPaths?: number; memoriesUsed?: number; }

export default function ChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hi! I'm your AI wellness companion. How are you feeling today?", timestamp: new Date() }
  ]);
  const [input, setInput]   = useState("");
  const [loading, setLoading] = useState(false);
  const [ending, setEnding]   = useState(false);
  const [countdown, setCountdown] = useState<number|null>(null);
  const [showGraph, setShowGraph] = useState(true);
  const [lastDebug, setLastDebug] = useState<Record<string,number>|null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Count down and then redirect to timeline
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) { router.push("/timeline"); return; }
    const t = setTimeout(() => setCountdown(c => (c ?? 1) - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown, router]);


  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages(m => [...m, { role: "user", content: text, timestamp: new Date() }]);
    setLoading(true);
    try {
      const { data } = await chatApi.send(text);
      setLastDebug(data.debug);
      setMessages(m => [...m, {
        role: "assistant", content: data.response, timestamp: new Date(),
        graphPaths: data.debug.layer4_graph_paths, memoriesUsed: data.memories_used
      }]);
    } catch {
      setMessages(m => [...m, { role: "assistant", content: "Sorry, I had trouble connecting. Please try again.", timestamp: new Date() }]);
    } finally { setLoading(false); }
  };

  const handleEndSession = async () => {
    if (ending) return;
    setEnding(true);
    try {
      await chatApi.endSession();
      setMessages([{
        role: "assistant",
        content: "✅ Session ended! Generating your Timeline summary... redirecting in 5 seconds.",
        timestamp: new Date()
      }]);
      setCountdown(5); // starts the redirect countdown
    } catch (e) {
      console.error(e);
      setMessages(m => [...m, { role: "assistant", content: "Failed to end session. Please try again.", timestamp: new Date() }]);
    } finally {
      setEnding(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };

  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>
        {/* Header */}
        <div className={styles.header}>
          <div>
            <h1 style={{fontSize:"18px",fontWeight:700}}>AI Wellness Chat</h1>
            <p className="text-muted" style={{fontSize:"13px"}}>Powered by Temporal GraphRAG + 3-Layer Memory</p>
          </div>
          <div style={{display:"flex",gap:"12px",alignItems:"center"}}>
            {lastDebug && (
              <div style={{display:"flex",gap:"8px"}}>
                <span className="badge badge-accent">{lastDebug.layer3_after_decay ?? 0} memories</span>
                <span className="badge badge-muted">{lastDebug.layer4_graph_paths ?? 0} graph paths</span>
                <span className="badge badge-muted">{lastDebug.elapsed_ms ?? 0}ms</span>
              </div>
            )}
            <button className="btn btn-ghost btn-sm" onClick={handleEndSession} disabled={ending || countdown !== null}>
              {ending ? <span className="spinner" style={{width:"14px",height:"14px",marginRight:"6px",display:"inline-block"}} /> : null}
              {countdown !== null ? `↗ Redirecting in ${countdown}s…` : ending ? "Ending..." : "End Session"}
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className={styles.messages}>
          {messages.map((msg, i) => (
            <div key={i} className={`${styles.msgRow} ${msg.role==="user" ? styles.userRow : ""} animate-fadeInUp`}
              style={{animationDelay:`${i*20}ms`}}>
              {msg.role === "assistant" && <div className={styles.avatar}>🤖</div>}
              <div className={`${styles.bubble} ${msg.role==="user" ? styles.userBubble : styles.aiBubble}`}>
                <p style={{whiteSpace:"pre-wrap",lineHeight:1.7}}>{msg.content}</p>
                <div style={{display:"flex",gap:"8px",marginTop:"6px",alignItems:"center"}}>
                  <span style={{fontSize:"11px",color:"var(--text-muted)"}}>
                    {msg.timestamp.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}
                  </span>
                  {msg.graphPaths != null && msg.graphPaths > 0 && (
                    <span className="badge badge-accent" style={{fontSize:"10px",padding:"2px 8px"}}>⬡ {msg.graphPaths} graph paths</span>
                  )}
                  {msg.memoriesUsed != null && msg.memoriesUsed > 0 && (
                    <span className="badge badge-muted" style={{fontSize:"10px",padding:"2px 8px"}}>🧠 {msg.memoriesUsed} memories</span>
                  )}
                </div>
              </div>
              {msg.role === "user" && <div className={styles.userAvatar}>👤</div>}
            </div>
          ))}

          {loading && (
            <div className={styles.msgRow}>
              <div className={styles.avatar}>🤖</div>
              <div className={styles.aiBubble} style={{display:"flex",gap:"6px",alignItems:"center",padding:"16px 20px"}}>
                <span className={styles.typingDot} style={{animationDelay:"0ms"}} />
                <span className={styles.typingDot} style={{animationDelay:"200ms"}} />
                <span className={styles.typingDot} style={{animationDelay:"400ms"}} />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className={styles.inputArea}>
          <textarea ref={inputRef} className={styles.textarea} placeholder="Share what's on your mind… (Enter to send, Shift+Enter for newline)"
            value={input} onChange={e=>setInput(e.target.value)} onKeyDown={handleKey} rows={1} disabled={loading} />
          <button className="btn btn-primary" onClick={send} disabled={loading||!input.trim()}>
            {loading ? <span className="spinner" style={{width:"16px",height:"16px"}} /> : "→"}
          </button>
        </div>
      </div>
    </div>
  );
}
