import React, { useState } from "react";
import { ExecutionStep } from "../types";

interface ToolExecutionPanelProps {
  steps: ExecutionStep[];
}

const TOOL_ICONS: Record<string, string> = {
  lookup_customer: "👤",
  calculate_base_credit_score: "📊",
  calculate_loan_adjusted_score: "🏦",
  calculate_dti_ratio: "📉",
  evaluate_collateral: "🏠",
  analyze_payment_history: "📅",
  compute_risk_weighted_score: "⚖️",
  ingest_document_data: "📄",
};

function JsonCollapsible({ data, label }: { data: unknown; label: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1"
      >
        <svg
          className={`w-3 h-3 transition-transform ${open ? "rotate-90" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        {label}
      </button>
      {open && (
        <pre className="mt-1 bg-gray-900 rounded p-2 overflow-x-auto text-green-400 text-xs max-h-40">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default function ToolExecutionPanel({ steps }: ToolExecutionPanelProps) {
  const totalMs = steps.reduce((sum, s) => sum + s.duration_ms, 0);

  if (steps.length === 0) {
    return (
      <div className="px-4 py-3 text-gray-500 text-xs">No tool calls yet.</div>
    );
  }

  return (
    <div className="px-4 py-3">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-gray-300">
          Tool Execution Trace
        </span>
        <span className="text-xs text-gray-500">
          Total: {totalMs.toFixed(0)}ms
        </span>
      </div>

      <div className="space-y-2">
        {steps.map((step, i) => (
          <React.Fragment key={i}>
            {/* Arrow connector */}
            {i > 0 && (
              <div className="flex items-center gap-2 py-0.5">
                <div className="w-6 flex justify-center">
                  <div className="w-px h-4 bg-gray-700" />
                </div>
                <svg className="w-3 h-3 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </div>
            )}

            <div className="flex items-start gap-3">
              {/* Step number */}
              <div className="w-6 h-6 rounded-full bg-gray-800 flex items-center justify-center text-xs text-gray-400 flex-shrink-0 mt-0.5">
                {i + 1}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm">
                      {TOOL_ICONS[step.tool_name] || "🔧"}
                    </span>
                    <span className="text-xs font-mono text-blue-400">
                      {step.tool_name}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-1.5 py-0.5 rounded ${
                        step.success
                          ? "bg-green-900 text-green-400"
                          : "bg-red-900 text-red-400"
                      }`}
                    >
                      {step.success ? "✓" : "✗"} {step.duration_ms.toFixed(0)}ms
                    </span>
                  </div>
                </div>

                {step.error && (
                  <p className="text-xs text-red-400 mb-1">{step.error}</p>
                )}

                <div className="space-y-1">
                  <JsonCollapsible data={step.tool_input} label="Input" />
                  {step.tool_output && (
                    <JsonCollapsible data={step.tool_output} label="Output" />
                  )}
                </div>
              </div>
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
