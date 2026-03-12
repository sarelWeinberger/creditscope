import React, { useState } from "react";
import { ExpertHeatmap, HeatmapCell, MoERequestTrace } from "../types";

interface MoEExpertPanelProps {
  requestTrace?: MoERequestTrace | null;
  fullView?: boolean;
  heatmapData?: ExpertHeatmap | null;
  autoRefresh?: boolean;
  onToggleAutoRefresh?: () => void;
}

function heatColor(freq: number): string {
  if (freq <= 0) return "#1f2937";
  if (freq < 0.1) return "#1e3a5f";
  if (freq < 0.25) return "#1e40af";
  if (freq < 0.40) return "#2563eb";
  if (freq < 0.60) return "#7c3aed";
  if (freq < 0.80) return "#a855f7";
  return "#ec4899";
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  content: string;
}

function HeatmapGrid({ heatmap }: { heatmap: ExpertHeatmap }) {
  const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0, content: "" });
  const { layers, experts, data } = heatmap;

  const maxExperts = 32;
  const displayExperts = experts.slice(0, maxExperts);
  const cellSize = 14;
  const labelW = 100;

  if (layers.length === 0) {
    return <p className="text-xs text-gray-500 px-4 py-2">No expert activation data yet.</p>;
  }

  return (
    <div className="relative px-4 py-2 overflow-x-auto">
      {tooltip.visible && (
        <div
          className="fixed z-50 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 pointer-events-none"
          style={{ left: tooltip.x + 8, top: tooltip.y - 40 }}
        >
          {tooltip.content}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: `${labelW}px repeat(${displayExperts.length}, ${cellSize}px)`, gap: 1 }}>
        {/* Header row: expert IDs */}
        <div />
        {displayExperts.map((eid, j) => (
          <div key={j} className="text-center text-gray-600" style={{ fontSize: 9, lineHeight: `${cellSize}px` }}>
            {j % 4 === 0 ? eid : ""}
          </div>
        ))}

        {/* Data rows */}
        {layers.map((layer, i) => (
          <React.Fragment key={layer}>
            <div className="text-right pr-2 text-gray-500 truncate" style={{ fontSize: 9, lineHeight: `${cellSize}px`, maxWidth: labelW }}>
              {layer.split(".").slice(-2).join(".")}
            </div>
            {displayExperts.map((eid, j) => {
              const cell: HeatmapCell = data[i]?.[j] || { count: 0, frequency: 0, avg_weight: 0 };
              return (
                <div
                  key={j}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: heatColor(cell.frequency),
                    borderRadius: 2,
                    cursor: "crosshair",
                  }}
                  onMouseEnter={(e) =>
                    setTooltip({
                      visible: true,
                      x: e.clientX,
                      y: e.clientY,
                      content: `Layer: ${layer}\nExpert ${eid}: ${cell.count} activations (${(cell.frequency * 100).toFixed(1)}%), avg weight ${cell.avg_weight.toFixed(3)}`,
                    })
                  }
                  onMouseLeave={() => setTooltip((t) => ({ ...t, visible: false }))}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 mt-3">
        <span className="text-xs text-gray-500">Low</span>
        {[0, 0.15, 0.35, 0.55, 0.75, 1.0].map((f) => (
          <div key={f} className="w-4 h-3 rounded-sm" style={{ backgroundColor: heatColor(f) }} />
        ))}
        <span className="text-xs text-gray-500">High</span>
      </div>
    </div>
  );
}

export default function MoEExpertPanel({ requestTrace, fullView = false, heatmapData = null, autoRefresh = false, onToggleAutoRefresh }: MoEExpertPanelProps) {
  const [view, setView] = useState<"request" | "session">("session");

  return (
    <div className={`${fullView ? "h-full" : ""} flex flex-col`}>
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-300">Expert Activation Heatmap</span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setView("session")}
            className={`text-xs px-2 py-0.5 rounded ${view === "session" ? "bg-purple-700 text-white" : "text-gray-500 hover:text-white"}`}
          >
            Session
          </button>
          <button
            onClick={() => setView("request")}
            className={`text-xs px-2 py-0.5 rounded ${view === "request" ? "bg-purple-700 text-white" : "text-gray-500 hover:text-white"}`}
          >
            This Request
          </button>
          {onToggleAutoRefresh && (
            <button
              onClick={onToggleAutoRefresh}
              className={`text-xs px-2 py-0.5 rounded border ${autoRefresh ? "border-green-600 text-green-400" : "border-gray-700 text-gray-500"}`}
            >
              {autoRefresh ? "● Live" : "Paused"}
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {heatmapData && view === "session" ? (
          <HeatmapGrid heatmap={heatmapData} />
        ) : (
          <div className="px-4 py-2">
            <p className="text-xs text-gray-500">
              {heatmapData === null
                ? "Waiting for expert data..."
                : view === "request" && !requestTrace
                ? "No trace for this request."
                : "Loading..."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
