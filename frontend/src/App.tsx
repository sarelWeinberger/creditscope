import React, { useCallback, useEffect, useState } from "react";
import { BrowserRouter, Route, Routes, Link, useLocation } from "react-router-dom";
import ChatInterface from "./components/ChatInterface";
import LoginScreen from "./components/LoginScreen";
import ObservabilityDash from "./components/ObservabilityDash";
import { Customer, CoTConfig, ThinkingMode, ThinkingVisibility } from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

function toNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function deriveCreditGrade(score: number): string {
  if (score >= 750) return "A";
  if (score >= 700) return "B";
  if (score >= 650) return "C";
  if (score >= 600) return "D";
  if (score >= 550) return "E";
  return "F";
}

function normalizeCustomer(raw: Record<string, unknown>): Customer {
  const annualIncome = toNumber(raw.annual_income);
  const monthlyIncome = annualIncome > 0 ? annualIncome / 12 : 0;
  const monthlyDebtPayments = toNumber(raw.monthly_debt_payments);
  const totalCreditLimit = toNumber(raw.total_credit_limit);
  const totalCreditUsed = toNumber(raw.total_credit_used);
  const creditScore = toNumber(raw.credit_score, toNumber(raw.fico_score, 300));
  const dtiRatio =
    toNumber(raw.dti_ratio, monthlyIncome > 0 ? monthlyDebtPayments / monthlyIncome : 0);
  const utilization =
    toNumber(
      raw.credit_utilization_ratio,
      totalCreditLimit > 0 ? totalCreditUsed / totalCreditLimit : 0,
    );

  return {
    id: toNumber(raw.id),
    first_name: String(raw.first_name || ""),
    last_name: String(raw.last_name || ""),
    full_name: String(raw.full_name || "Unknown Customer"),
    email: String(raw.email || ""),
    phone: typeof raw.phone === "string" ? raw.phone : undefined,
    date_of_birth: typeof raw.date_of_birth === "string" ? raw.date_of_birth : undefined,
    ssn_last4: typeof raw.ssn_last4 === "string" ? raw.ssn_last4 : undefined,
    address: typeof raw.address === "string" ? raw.address : undefined,
    city: typeof raw.city === "string" ? raw.city : undefined,
    state: typeof raw.state === "string" ? raw.state : undefined,
    zip_code: typeof raw.zip_code === "string" ? raw.zip_code : undefined,
    employment_status: (String(raw.employment_status || "employed") as Customer["employment_status"]),
    employer_name: typeof raw.employer_name === "string" ? raw.employer_name : undefined,
    annual_income: annualIncome,
    monthly_income: toNumber(raw.monthly_income, monthlyIncome),
    years_employed: toNumber(raw.years_employed, toNumber(raw.years_at_current_job)),
    job_title: typeof raw.job_title === "string" ? raw.job_title : undefined,
    credit_score: creditScore,
    credit_history_years: toNumber(raw.credit_history_years),
    num_credit_accounts: toNumber(raw.num_credit_accounts, toNumber(raw.num_credit_cards)),
    num_open_accounts: toNumber(raw.num_open_accounts),
    num_closed_accounts: toNumber(raw.num_closed_accounts),
    total_credit_limit: totalCreditLimit,
    total_credit_used: totalCreditUsed,
    credit_utilization_ratio: utilization,
    credit_grade: String(raw.credit_grade || deriveCreditGrade(creditScore)),
    total_debt: toNumber(raw.total_debt, toNumber(raw.total_revolving_debt) + monthlyDebtPayments * 12),
    monthly_debt_payments: monthlyDebtPayments,
    mortgage_balance: toNumber(raw.mortgage_balance),
    auto_loan_balance: toNumber(raw.auto_loan_balance),
    student_loan_balance: toNumber(raw.student_loan_balance),
    credit_card_balance: toNumber(raw.credit_card_balance, totalCreditUsed),
    other_debt: toNumber(raw.other_debt),
    dti_ratio: dtiRatio,
    on_time_payments: toNumber(raw.on_time_payments),
    late_payments_30d: toNumber(raw.late_payments_30d, toNumber(raw.num_late_payments_12m)),
    late_payments_60d: toNumber(raw.late_payments_60d),
    late_payments_90d: toNumber(raw.late_payments_90d),
    collections: toNumber(raw.collections, toNumber(raw.num_collections)),
    bankruptcies: toNumber(raw.bankruptcies, toNumber(raw.num_bankruptcies)),
    foreclosures: toNumber(raw.foreclosures),
    charge_offs: toNumber(raw.charge_offs, toNumber(raw.num_defaults)),
    hard_inquiries_6m: toNumber(raw.hard_inquiries_6m, toNumber(raw.num_hard_inquiries_6m)),
    hard_inquiries_12m: toNumber(raw.hard_inquiries_12m, toNumber(raw.num_hard_inquiries_12m)),
    soft_inquiries_12m: toNumber(raw.soft_inquiries_12m),
    checking_balance: toNumber(raw.checking_balance),
    savings_balance: toNumber(raw.savings_balance),
    investment_balance: toNumber(raw.investment_balance),
    property_value: toNumber(raw.property_value),
    vehicle_value: toNumber(raw.vehicle_value),
    is_active: raw.is_active === undefined ? true : Boolean(raw.is_active),
    created_at: String(raw.created_at || ""),
    updated_at: String(raw.updated_at || ""),
    notes: typeof raw.notes === "string" ? raw.notes : undefined,
  };
}

// ─── Customer List Page ───────────────────────────────────────────────────────

function CustomersPage({ onSelect }: { onSelect: (c: Customer) => void }) {
  const [customers, setCustomers] = React.useState<Customer[]>([]);
  const [search, setSearch] = React.useState("");
  const [debouncedSearch, setDebouncedSearch] = React.useState("");
  const [loading, setLoading] = React.useState(false);

  // Debounce search input by 300ms
  React.useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  React.useEffect(() => {
    setLoading(true);
    const params = debouncedSearch ? `?search=${encodeURIComponent(debouncedSearch)}&search_type=fuzzy` : "?page=1&page_size=30";
    fetch(`${API_BASE}/customers${params}`, { credentials: "include" })
      .then((r) => r.json())
      .then((d) => setCustomers((d.customers || []).map((customer: Record<string, unknown>) => normalizeCustomer(customer))))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [debouncedSearch]);

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

function AppLayout({ currentUser, onLogout }: { currentUser: string; onLogout: () => Promise<void> }) {
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
          <div className="mt-4 rounded-lg border border-gray-800 bg-gray-950/80 px-3 py-2">
            <p className="truncate text-xs text-gray-400">Signed in as</p>
            <p className="truncate text-sm text-white">{currentUser}</p>
            <button
              type="button"
              onClick={() => {
                void onLogout();
              }}
              className="mt-2 text-xs font-medium text-blue-400 hover:text-blue-300"
            >
              Sign out
            </button>
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
  const [authStatus, setAuthStatus] = useState<"checking" | "authenticated" | "unauthenticated">("checking");
  const [currentUser, setCurrentUser] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  const refreshSession = useCallback(async () => {
    setAuthStatus("checking");
    try {
      const response = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
      if (!response.ok) {
        throw new Error("Unauthenticated");
      }

      const data = (await response.json()) as { email: string };
      setCurrentUser(data.email);
      setAuthError(null);
      setAuthStatus("authenticated");
    } catch {
      setCurrentUser("");
      setAuthStatus("unauthenticated");
    }
  }, []);

  useEffect(() => {
    void refreshSession();
  }, [refreshSession]);

  const handleLogin = useCallback(async (email: string, password: string) => {
    setIsAuthenticating(true);
    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: string } | null;
        setAuthError(payload?.detail || "Invalid email or password");
        setAuthStatus("unauthenticated");
        return;
      }

      const data = (await response.json()) as { email: string };
      setCurrentUser(data.email);
      setAuthError(null);
      setAuthStatus("authenticated");
    } finally {
      setIsAuthenticating(false);
    }
  }, []);

  const handleLogout = useCallback(async () => {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    }).catch(() => undefined);
    setCurrentUser("");
    setAuthError(null);
    setAuthStatus("unauthenticated");
  }, []);

  if (authStatus === "checking") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-950 text-gray-100">
        <div className="rounded-2xl border border-gray-800 bg-gray-900 px-6 py-5 text-sm text-gray-300">
          Checking session...
        </div>
      </div>
    );
  }

  if (authStatus !== "authenticated") {
    return (
      <LoginScreen
        error={authError}
        isSubmitting={isAuthenticating}
        onLogin={handleLogin}
      />
    );
  }

  return (
    <BrowserRouter>
      <AppLayout currentUser={currentUser} onLogout={handleLogout} />
    </BrowserRouter>
  );
}
