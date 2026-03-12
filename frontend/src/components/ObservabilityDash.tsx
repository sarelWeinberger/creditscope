import React from "react";
import { useObservability } from "../hooks/useObservability";
import MoEExpertPanel from "./MoEExpertPanel";
import LayerActivityView from "./LayerActivityView";
import { EntropyDataPoint } from "../types";

// Simple SVG line chart for entropy time series
function EntropyChart({ data }: { data: EntropyDataPoint[] }) {
  const W = 600;
  const H = 120;
  const PAD = { top: 10, right: 10, bottom: 25, left: 40 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;

  if (data.length < 2) {
    return (
      <div
        className="flex items-center justify-center text-gray-600 text-xs"
        style={{ width: W, height: H }}
      >
        Collecting entropy data...
      </div>
    );
  }

  const layers = [...new Set(data.map((d) => d.layer))].slice(0, 5);
  const colors = ["#7c3aed", "#2563eb", "#0891b2", "#059669", "#d97706"];

  const times = data.map((d) => d.timestamp);
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const entropies = data.map((d) => d.entropy);
  const eMin = 0;
  const eMax = Math.max(...entropies, 3);

  const scaleX = (t: number) => PAD.left + ((t - tMin) / (tMax - tMin || 1)) * IW;
  const scaleY = (e: number) => PAD.top + IH - ((e - eMin) / (eMax - eMin || 1)) * IH;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
      {/* Y grid lines */}
      {[0, 1, 2, 3].map((v) => (
        <g key={v}>
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={scaleY(v)}
            y2={scaleY(v)}
            stroke="#374151"
            strokeDasharray="4,4"
          />
          <text x={PAD.left - 4} y={scaleY(v) + 4} fontSize={9} textAnchor="end" fill="#6b7280">
            {v}
          </text>
        </g>
      ))}

      {/* X axis label */}
      <text x={W / 2} y={H - 2} fontSize={9} textAnchor="middle" fill="#6b7280">
        Time →
      </text>

      {/* Line per layer */}
      {layers.map((layer, li) => {
        const layerData = data.filter((d) => d.layer === layer);
        if (layerData.length < 2) return null;
        const points = layerData
          .map((d) => `${scaleX(d.timestamp)},${scaleY(d.entropy)}`)
          .join(" ");
        return (
          <polyline
            key={layer}
            points={points}
            fill="none"
            stroke={colors[li % colors.length]}
            strokeWidth={1.5}
            strokeLinejoin="round"
            opacity={0.9}
          />
        );
      })}

      {/* Legend */}
      {layers.map((layer, li) => (
        <g key={layer} transform={`translate(${PAD.left + li * 110}, ${H - 10})`}>
          <line x1={0} x2={12} y1={0} y2={0} stroke={colors[li % colors.length]} strokeWidth={2} />
          <text x={15} y={4} fontSize={8} fill="#9ca3af">
            {layer.split(".").slice(-1)[0]}
          </text>
        </g>
      ))}
    </svg>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function ObservabilityDash() {
  const {
    heatmapData,
    entropyData,
    layerData,
    isLoading,
    lastUpdated,
    autoRefresh,
    setAutoRefresh,
    refresh,
  } = useObservability();

  const totalActivations = heatmapData?.data
    .flat()
    .reduce((s, c) => s + c.count, 0) || 0;

  const avgEntropy =
    entropyData.length > 0
      ? (entropyData.reduce((s, d) => s + d.entropy, 0) / entropyData.length).toFixed(3)
      : "—";

  const topLayer = layerData[0];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">MoE Observability Dashboard</h2>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-gray-500">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1 text-xs rounded-md border transition-colors ${
              autoRefresh
                ? "border-green-600 text-green-400 bg-green-950"
                : "border-gray-700 text-gray-400"
            }`}
          >
            {autoRefresh ? "● Live" : "Paused"}
          </button>
          <button
            onClick={refresh}
            disabled={isLoading}
            className="px-3 py-1 text-xs rounded-md border border-gray-700 text-gray-400 hover:text-white disabled:opacity-40 transition-colors"
          >
            {isLoading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Expert Activations"
          value={totalActivations.toLocaleString()}
          sub={`${heatmapData?.total_requests || 0} requests`}
        />
        <MetricCard
          label="Avg Router Entropy"
          value={avgEntropy}
          sub="Higher = more balanced routing"
        />
        <MetricCard
          label="Active Layers"
          value={String(layerData.length)}
          sub={`${heatmapData?.experts.length || 0} experts`}
        />
        <MetricCard
          label="Top Layer"
          value={topLayer ? topLayer.activation_count.toLocaleString() : "—"}
          sub={topLayer ? topLayer.layer_id.split(".").slice(-2).join(".") : "No data"}
        />
      </div>

      {/* Expert heatmap */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">
          Expert Activation Heatmap
        </h3>
        <MoEExpertPanel
          heatmapData={heatmapData}
          autoRefresh={autoRefresh}
          onToggleAutoRefresh={() => setAutoRefresh(!autoRefresh)}
        />
      </div>

      {/* Entropy chart */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">
          Router Entropy Over Time
        </h3>
        <EntropyChart data={entropyData} />
      </div>

      {/* Layer activity */}
      <div className="bg-gray-900 rounded-xl border border-gray-800">
        <h3 className="text-sm font-semibold text-gray-300 p-4 border-b border-gray-800">
          Per-Layer Expert Activity
        </h3>
        <LayerActivityView layers={layerData} />
      </div>
    </div>
  );
}
