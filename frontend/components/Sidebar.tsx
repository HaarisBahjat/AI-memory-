"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuthStore } from "@/store/auth";
import { authApi, userApi } from "@/lib/api";
import styles from "./Sidebar.module.css";

const NAV = [
  { href: "/chat",     icon: "💬", label: "Chat" },
  { href: "/memories", icon: "🧠", label: "Memories" },
  { href: "/timeline", icon: "📈", label: "Timeline" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router   = useRouter();
  const { logout, refreshToken } = useAuthStore();
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    userApi.getProfile().then(r => setIsAdmin(r.data.is_admin)).catch(() => {});
  }, []);

  const handleLogout = async () => {
    try { if (refreshToken) await authApi.logout(refreshToken); } catch {}
    logout();
    router.push("/login");
  };

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <span className="gradient-text" style={{fontSize:"20px",fontWeight:800}}>⬡ AMS</span>
        <span className={styles.logoSub}>Memory System</span>
      </div>

      <nav className={styles.nav}>
        {NAV.map(({ href, icon, label }) => (
          <Link key={href} href={href}
            className={`${styles.navItem} ${pathname.startsWith(href) ? styles.active : ""}`}>
            <span className={styles.navIcon}>{icon}</span>
            <span>{label}</span>
          </Link>
        ))}
        {isAdmin && (
          <Link href="/admin"
            className={`${styles.navItem} ${pathname.startsWith("/admin") ? styles.active : ""}`}>
            <span className={styles.navIcon}>⚙️</span>
            <span>Admin</span>
          </Link>
        )}
      </nav>

      <button onClick={handleLogout} className={`btn btn-ghost btn-sm ${styles.logout}`}>
        ← Logout
      </button>
    </aside>
  );
}
