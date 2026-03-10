import React, { useState } from "react";
import { BrowserRouter, Route, Routes, Link, useLocation } from "react-router-dom";
import ChatInterface from "./components/ChatInterface";
import ObservabilityDash from "./components/ObservabilityDash";
import { Customer, CoTConfig, ThinkingMode, ThinkingVisibility } from "./types";

// ─── Customer List Page ───────────────────────────────────────────────────────

function CustomersPage({ onSelect }: { onSelect: (c: Customer) => void }) {
  const [customers, setCustomers] = React.useState<Customer[]>([]);
  const [search, setSearch] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const API = import.meta.env.VITE_API_URL || "http://localhost:8080/api";

  React.useEffect(() => {
    setLoading(true);
    const params = search ? `?search=${encodeURIComponent(search)}&search_type=fuzzy` : "?page=1&page_size=30";
    fetch(`${API}/customers${params}`)
      .then((r) => r.json())
      .then((d) => setCustomers(d.customers || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [search]);

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Customers</h1>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by name, email..."
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500 w-72"
        />
      </div>
      {loading ? (
        <div className="flex items-center justify-center h-40 text-gray-500">Loading...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {customers.map((c) => (
            <div
              key={c.id}
              onClick={() => onSelect(c)}
              className="bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-xl p-4 cursor-pointer transition-all"
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="text-white font-semibold">{c.full_name}</h3>
                  <p className="text-gray-400 text-xs">{c.job_title || c.employment_status}</p>
                </div>
                <div className="text-right">
                  <span className="text-2xl font-bold text-white">{c.credit_score}</span>
                  <p className={`text-xs font-medium ${
                    c.credit_grade === "A" ? "text-green-400" :
                    c.credit_grade === "B" ? "text-lime-400" :
                    c.credit_grade === "C" ? "text-yellow-400" :
                    c.credit_grade === "D" ? "text-orange-400" : "text-red-400"
                  }`}>Grade {c.credit_grade}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs text-center mt-3">
                <div className="bg-gray-800 rounded p-1.5">
                  <p className="text-gray-500">Income</p>
                  <p className="text-gray-200 font-medium">
                    ${(c.annual_income / 1000).toFixed(0)}K
                  </p>
                </div>
                <div className="bg-gray-800 rounded p-1.5">
                  <p className="text-gray-500">DTI</p>
                  <p className="text-gray-200 font-medium">
                    {(c.dti_ratio * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-800 rounded p-1.5">
                  <p className="text-gray-500">Util</p>
                  <p className="text-gray-200 font-medium">
                    {(c.credit_utilization_ratio * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Thinking Config Page ─────────────────────────────────────────────────────

function ThinkingPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-white mb-6">Thinking Configuration</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          { name: "none", tokens: 0, desc: "Direct response, no reasoning pass", latency: "~0.1s", color: "gray" },
          { name: "minimal", tokens: 256, desc: "Quick sanity check reasoning", latency: "~1s", color: "blue" },
          { name: "short", tokens: 512, desc: "Brief reasoning for simple queries", latency: "~2s", color: "blue" },
          { name: "standard", tokens: 1024, desc: "Balanced analysis for most credit queries", latency: "~4s", color: "purple" },
          { name: "extended", tokens: 2048, desc: "Thorough multi-factor reasoning", latency: "~8s", color: "purple" },
          { name: "deep", tokens: 4096, desc: "Complex analysis, regulatory review", latency: "~16s", color: "pink" },
          { name: "unlimited", tokens: -1, desc: "Full extended thinking, no budget cap", latency: "30s+", color: "red" },
        ].map((preset) => (
          <div key={preset.name} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-white font-semibold capitalize">{preset.name}</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">{preset.latency}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
                  {preset.tokens === -1 ? "∞" : `${preset.tokens} tok`}
                </span>
              </div>
            </div>
            <p className="text-sm text-gray-400">{preset.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Navigation ───────────────────────────────────────────────────────────────

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  const { pathname } = useLocation();
  const active = pathname === to || (to !== "/" && pathname.startsWith(to));
  return (
    <Link
      to={to}
      className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
        active ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white hover:bg-gray-800"
      }`}
    >
      {icon}
      {label}
    </Link>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

function AppLayout() {
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 border-r border-gray-800 bg-gray-900 flex flex-col">
        {/* Brand */}
        <div className="px-4 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xs">
              CS
            </div>
            <div>
              <p className="text-white font-semibold text-sm">CreditScope</p>
              <p className="text-gray-500 text-xs">AI Credit Analysis</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <NavItem
            to="/"
            label="Chat"
            icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            }
          />
          <NavItem
            to="/customers"
            label="Customers"
            icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            }
          />
          <NavItem
            to="/observability"
            label="MoE Observability"
            icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            }
          />
          <NavItem
            to="/thinking"
            label="Thinking Config"
            icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            }
          />
        </nav>

        {/* Selected customer indicator */}
        {selectedCustomer && (
          <div className="px-3 py-3 border-t border-gray-800">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <p className="text-xs text-gray-500">Active Customer</p>
                <p className="text-sm text-white font-medium truncate">{selectedCustomer.full_name}</p>
              </div>
              <button
                onClick={() => setSelectedCustomer(null)}
                className="text-gray-500 hover:text-gray-300 flex-shrink-0"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatInterface selectedCustomer={selectedCustomer} />} />
          <Route
            path="/customers"
            element={
              <div className="h-full overflow-y-auto">
                <CustomersPage onSelect={setSelectedCustomer} />
              </div>
            }
          />
          <Route
            path="/observability"
            element={
              <div className="h-full overflow-y-auto">
                <ObservabilityDash />
              </div>
            }
          />
          <Route
            path="/thinking"
            element={
              <div className="h-full overflow-y-auto">
                <ThinkingPage />
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
