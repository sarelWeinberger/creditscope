import React, { useState, useEffect, useRef } from "react";
import { ThinkingVisibility } from "../types";

interface ThinkingPanelProps {
  content: string;
  tokensUsed?: number;
  durationMs?: number;
  wasEnforced?: boolean;
  isStreaming?: boolean;
  visibility?: ThinkingVisibility;
  thinkingPhaseExperts?: number[];
  responsePhaseExperts?: number[];
}

function BudgetBar({ used, budget }: { used: number; budget: number }) {
  if (!budget || budget === -1) return null;
  const pct = Math.min((used / budget) * 100, 100);
  const color = pct < 80 ? "bg-green-500" : pct < 95 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500">{pct.toFixed(0)}%</span>
    </div>
  );
}

function TypewriterText({ text, active }: { text: string; active: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [text]);

  return (
    <div
      ref={ref}
      className="text-xs text-gray-400 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto leading-relaxed"
    >
      {text}
      {active && <span className="animate-pulse text-purple-400">▌</span>}
    </div>
  );
}

export default function ThinkingPanel({
  content,
  tokensUsed = 0,
  durationMs,
  wasEnforced = false,
  isStreaming = false,
  visibility = ThinkingVisibility.Collapsed,
  thinkingPhaseExperts,
  responsePhaseExperts,
}: ThinkingPanelProps) {
  const [mode, setMode] = useState<"collapsed" | "streaming" | "full">(
    visibility === ThinkingVisibility.Full
      ? "full"
      : visibility === ThinkingVisibility.Streaming
      ? "streaming"
      : "collapsed"
  );

  if (!content && !isStreaming) return null;
  if (visibility === ThinkingVisibility.Hidden) return null;

  const showContent = mode !== "collapsed" || isStreaming;
  const maxBudget = 2048; // default — could be passed as prop

  return (
    <div
      className={`rounded-xl border bg-gray-900 ${
        wasEnforced ? "border-orange-700" : "border-purple-900"
      } max-w-[85%]`}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2 cursor-pointer select-none"
        onClick={() => setMode((m) => (m === "collapsed" ? "full" : "collapsed"))}
      >
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isStreaming ? "bg-purple-400 animate-pulse" : "bg-purple-600"}`} />
          <span className="text-xs font-medium text-purple-300">
            {isStreaming ? "Thinking..." : "Thought process"}
          </span>
          {tokensUsed > 0 && (
            <span className="text-xs text-gray-500">{tokensUsed} tokens</span>
          )}
          {durationMs && (
            <span className="text-xs text-gray-500">{(durationMs / 1000).toFixed(1)}s</span>
          )}
          {wasEnforced && (
            <span className="text-xs px-1.5 py-0.5 bg-orange-900 text-orange-400 rounded">
              Budget enforced
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => { e.stopPropagation(); setMode("streaming"); }}
            className={`text-xs px-1.5 py-0.5 rounded ${mode === "streaming" ? "bg-purple-800 text-white" : "text-gray-600 hover:text-gray-400"}`}
          >
            Stream
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); setMode("full"); }}
            className={`text-xs px-1.5 py-0.5 rounded ${mode === "full" ? "bg-purple-800 text-white" : "text-gray-600 hover:text-gray-400"}`}
          >
            Full
          </button>
          <svg
            className={`w-3 h-3 text-gray-500 transition-transform ${showContent ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {showContent && (
        <div className="px-3 pb-3 border-t border-gray-800">
          <div className="mt-2">
            {mode === "streaming" ? (
              <TypewriterText text={content} active={isStreaming} />
            ) : (
              <div className="text-xs text-gray-400 font-mono whitespace-pre-wrap max-h-60 overflow-y-auto leading-relaxed">
                {content || (isStreaming ? <span className="animate-pulse text-purple-400">▌</span> : "—")}
              </div>
            )}
          </div>

          {/* Budget bar */}
          {tokensUsed > 0 && (
            <div className="mt-2">
              <BudgetBar used={tokensUsed} budget={maxBudget} />
            </div>
          )}

          {/* Expert phase comparison */}
          {thinkingPhaseExperts && responsePhaseExperts && (
            <div className="mt-3 pt-2 border-t border-gray-800">
              <p className="text-xs text-gray-500 mb-2">Expert Phase Comparison</p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs text-purple-400 mb-1">Thinking ({thinkingPhaseExperts.length})</p>
                  <div className="flex flex-wrap gap-1">
                    {thinkingPhaseExperts.slice(0, 12).map((e) => (
                      <span key={e} className="text-xs bg-purple-900 text-purple-300 px-1 rounded">
                        E{e}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-blue-400 mb-1">Response ({responsePhaseExperts.length})</p>
                  <div className="flex flex-wrap gap-1">
                    {responsePhaseExperts.slice(0, 12).map((e) => (
                      <span key={e} className="text-xs bg-blue-900 text-blue-300 px-1 rounded">
                        E{e}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
