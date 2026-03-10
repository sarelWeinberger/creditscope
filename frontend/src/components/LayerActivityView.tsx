import React, { useState } from "react";
import { LayerActivity } from "../types";

interface LayerActivityViewProps {
  layers: LayerActivity[];
}

function EntropyGauge({ entropy, maxEntropy = 3 }: { entropy: number; maxEntropy?: number }) {
  const pct = Math.min(entropy / maxEntropy, 1);
  const color = pct < 0.4 ? "#ef4444" : pct < 0.7 ? "#eab308" : "#22c55e";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs text-gray-400 w-12 text-right">{entropy.toFixed(3)}</span>
    </div>
  );
}

function ExpertBar({ expert, maxCount }: { expert: LayerActivity["top_experts"][0]; maxCount: number }) {
  const pct = maxCount > 0 ? (expert.count / maxCount) * 100 : 0;
  return (
    <div className="flex items-center gap-2 mb-1">
      <span className="text-xs text-gray-500 w-16 text-right">E{expert.expert_id}</span>
      <div className="flex-1 h-3 bg-gray-800 rounded overflow-hidden">
        <div
          className="h-full rounded"
          style={{
            width: `${pct}%`,
            backgroundColor: `hsl(${260 + pct * 0.5}, 70%, 60%)`,
          }}
        />
      </div>
      <span className="text-xs text-gray-400 w-16">{expert.count} ({(expert.frequency * 100).toFixed(1)}%)</span>
    </div>
  );
}

function LayerRow({ layer }: { layer: LayerActivity }) {
  const [expanded, setExpanded] = useState(false);
  const maxCount = Math.max(...layer.top_experts.map((e) => e.count), 1);

  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <svg
            className={`w-3 h-3 text-gray-500 flex-shrink-0 transition-transform ${expanded ? "rotate-90" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <span className="text-xs font-mono text-gray-300 truncate">{layer.layer_id}</span>
        </div>
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-500">Entropy:</span>
            <span className="text-xs text-yellow-400">{layer.entropy.toFixed(3)}</span>
          </div>
          <span className="text-xs text-gray-500">{layer.activation_count} acts</span>
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 border-t border-gray-800 bg-gray-900">
          {/* Entropy gauge */}
          <div className="mt-2 mb-3">
            <p className="text-xs text-gray-500 mb-1">Router Entropy (higher = more balanced)</p>
            <EntropyGauge entropy={layer.entropy} />
          </div>

          {/* Top experts */}
          <p className="text-xs text-gray-500 mb-2">Top Expert Activations</p>
          {layer.top_experts.map((expert) => (
            <ExpertBar key={expert.expert_id} expert={expert} maxCount={maxCount} />
          ))}

          {/* Weight stats */}
          {Object.keys(layer.gating_weight_stats).length > 0 && (
            <div className="mt-3 grid grid-cols-3 gap-2">
              {Object.entries(layer.gating_weight_stats).map(([k, v]) => (
                <div key={k} className="text-center bg-gray-800 rounded p-1.5">
                  <p className="text-xs text-gray-500 capitalize">{k}</p>
                  <p className="text-xs font-medium text-gray-200">{v.toFixed(4)}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function LayerActivityView({ layers }: LayerActivityViewProps) {
  if (layers.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
        No layer activity data available.
      </div>
    );
  }

  return (
    <div className="space-y-2 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-gray-300">Layer Activity</span>
        <span className="text-xs text-gray-500">{layers.length} layers</span>
      </div>
      {layers.map((layer) => (
        <LayerRow key={layer.layer_id} layer={layer} />
      ))}
    </div>
  );
}
