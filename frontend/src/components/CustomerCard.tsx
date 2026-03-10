import React from "react";
import { Customer, LoanApplication } from "../types";
import CreditScoreGauge from "./CreditScoreGauge";

interface CustomerCardProps {
  customer: Customer;
  loans?: LoanApplication[];
  compact?: boolean;
  onClick?: () => void;
  selected?: boolean;
}

function GradeTag({ grade }: { grade: string }) {
  const colors: Record<string, string> = {
    A: "bg-green-900 text-green-300 border-green-700",
    B: "bg-teal-900 text-teal-300 border-teal-700",
    C: "bg-yellow-900 text-yellow-300 border-yellow-700",
    D: "bg-orange-900 text-orange-300 border-orange-700",
    E: "bg-red-900 text-red-300 border-red-700",
    F: "bg-red-950 text-red-400 border-red-800",
  };
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-xs font-bold rounded border ${
        colors[grade] || "bg-gray-800 text-gray-400 border-gray-600"
      }`}
    >
      Grade {grade}
    </span>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-gray-800 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-xs font-medium text-gray-200">{value}</span>
    </div>
  );
}

function formatCurrency(n: number) {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

function formatPct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

export default function CustomerCard({
  customer,
  loans,
  compact = false,
  onClick,
  selected = false,
}: CustomerCardProps) {
  const grade = customer.credit_grade || "C";

  if (compact) {
    return (
      <div className="space-y-4">
        <div>
          <h3 className="text-white font-semibold text-base">{customer.full_name}</h3>
          <p className="text-gray-400 text-xs">{customer.job_title || customer.employment_status}</p>
          {customer.employer_name && (
            <p className="text-gray-500 text-xs">{customer.employer_name}</p>
          )}
        </div>

        <CreditScoreGauge score={customer.credit_score} grade={grade} size="sm" />

        <div className="space-y-0.5">
          <MetricRow label="Annual Income" value={formatCurrency(customer.annual_income)} />
          <MetricRow label="Monthly Debt" value={formatCurrency(customer.monthly_debt_payments)} />
          <MetricRow label="DTI Ratio" value={formatPct(customer.dti_ratio)} />
          <MetricRow label="Credit Limit" value={formatCurrency(customer.total_credit_limit)} />
          <MetricRow label="Utilization" value={formatPct(customer.credit_utilization_ratio)} />
          <MetricRow label="On-time Payments" value={String(customer.on_time_payments)} />
          <MetricRow label="Late 30d/60d/90d" value={`${customer.late_payments_30d}/${customer.late_payments_60d}/${customer.late_payments_90d}`} />
          {customer.bankruptcies > 0 && (
            <MetricRow label="Bankruptcies" value={String(customer.bankruptcies)} />
          )}
        </div>

        {loans && loans.length > 0 && (
          <div>
            <p className="text-xs text-gray-400 mb-2 font-medium">Loan Applications</p>
            <div className="space-y-1">
              {loans.slice(0, 3).map((loan) => (
                <div
                  key={loan.id}
                  className="flex items-center justify-between text-xs p-2 rounded bg-gray-800"
                >
                  <span className="text-gray-300 capitalize">{loan.loan_type}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400">{formatCurrency(loan.requested_amount)}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs ${
                        loan.status === "approved" || loan.status === "disbursed"
                          ? "bg-green-900 text-green-400"
                          : loan.status === "rejected"
                          ? "bg-red-900 text-red-400"
                          : "bg-gray-700 text-gray-400"
                      }`}
                    >
                      {loan.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Full card for list view
  return (
    <div
      onClick={onClick}
      className={`bg-gray-900 rounded-xl border transition-all cursor-pointer p-4 ${
        selected ? "border-blue-500 ring-1 ring-blue-500" : "border-gray-800 hover:border-gray-700"
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-white font-semibold">{customer.full_name}</h3>
          <p className="text-gray-400 text-xs mt-0.5">
            {customer.job_title || "—"} · {customer.city}, {customer.state}
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <GradeTag grade={grade} />
          <span className="text-lg font-bold text-white">{customer.credit_score}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="bg-gray-800 rounded-lg p-2">
          <p className="text-xs text-gray-500">Income</p>
          <p className="text-sm font-semibold text-white">{formatCurrency(customer.annual_income)}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-2">
          <p className="text-xs text-gray-500">DTI</p>
          <p className="text-sm font-semibold text-white">{formatPct(customer.dti_ratio)}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-2">
          <p className="text-xs text-gray-500">Utilization</p>
          <p className="text-sm font-semibold text-white">{formatPct(customer.credit_utilization_ratio)}</p>
        </div>
      </div>
    </div>
  );
}
