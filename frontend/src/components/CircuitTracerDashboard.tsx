import React, { useCallback, useState } from "react";
import {
  CircuitArchitecture,
  CircuitCheckpoint,
  CircuitTraceResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

async function readJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: "include",
    ...init,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-white">{value}</p>
      {sub ? <p className="mt-1 text-xs text-gray-500">{sub}</p> : null}
    </div>
  );
}

export default function CircuitTracerDashboard() {
  const [architecture, setArchitecture] = useState<CircuitArchitecture | null>(null);
  const [checkpoints, setCheckpoints] = useState<CircuitCheckpoint[]>([]);
  const [traceResult, setTraceResult] = useState<CircuitTraceResponse | null>(null);
  const [prompt, setPrompt] = useState("Analyze the applicant's risk factors for this loan decision.");
  const [keepFraction, setKeepFraction] = useState("0.10");
  const [prune, setPrune] = useState(true);
  const [loadingArchitecture, setLoadingArchitecture] = useState(false);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false);
  const [tracing, setTracing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadArchitecture = useCallback(async () => {
    setLoadingArchitecture(true);
    setError(null);
    try {
      const data = await readJson<CircuitArchitecture>(`${API_BASE}/circuit/architecture`);
      setArchitecture(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load architecture.");
    } finally {
      setLoadingArchitecture(false);
    }
  }, []);

  const loadCheckpoints = useCallback(async () => {
    setLoadingCheckpoints(true);
    setError(null);
    try {
      const data = await readJson<{ checkpoints: CircuitCheckpoint[] }>(`${API_BASE}/circuit/saes`);
      setCheckpoints(data.checkpoints || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load checkpoints.");
    } finally {
      setLoadingCheckpoints(false);
    }
  }, []);

  const runTrace = useCallback(async () => {
    setTracing(true);
    setError(null);
    try {
      const keep = Number.parseFloat(keepFraction);
      const payload = {
        prompt,
        prune,
        keep_fraction: Number.isFinite(keep) ? keep : 0.1,
      };
      const data = await readJson<CircuitTraceResponse>(`${API_BASE}/circuit/trace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setTraceResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Trace failed.");
    } finally {
      setTracing(false);
    }
  }, [keepFraction, prompt, prune]);

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Circuit Tracer</h1>
          <p className="mt-1 max-w-3xl text-sm text-gray-400">
            On-demand interpretability tools for architecture inspection, checkpoint discovery,
            and prompt tracing. Architecture and trace actions may load a separate local model
            and can consume substantial GPU memory.
          </p>
        </div>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-900 bg-red-950/50 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <StatCard
          label="Checkpoints"
          value={String(checkpoints.length)}
          sub={checkpoints.length > 0 ? "Available SAE or CLT files" : "No saved checkpoints yet"}
        />
        <StatCard
          label="Architecture"
          value={architecture ? `${architecture.num_layers} layers` : "Not loaded"}
          sub={architecture ? `${architecture.moe_layers.length} MoE, ${architecture.deltanet_layers.length} DeltaNet` : "Load on demand"}
        />
        <StatCard
          label="Latest Trace"
          value={traceResult ? `${traceResult.num_nodes} nodes` : "No trace"}
          sub={traceResult ? `${traceResult.num_edges} edges across ${traceResult.num_layers} layers` : "Run a trace to populate"}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr,0.8fr]">
        <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-300">Trace Prompt</h2>
            <button
              onClick={runTrace}
              disabled={tracing || !prompt.trim()}
              className="rounded-md border border-blue-600 bg-blue-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {tracing ? "Tracing..." : "Run Trace"}
            </button>
          </div>

          <div className="space-y-4">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              className="w-full rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none transition-colors focus:border-blue-500"
              placeholder="Enter a credit analysis prompt to trace..."
            />

            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={prune}
                  onChange={(e) => setPrune(e.target.checked)}
                  className="rounded border-gray-700 bg-gray-950"
                />
                Prune graph
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-300">
                Keep fraction
                <input
                  value={keepFraction}
                  onChange={(e) => setKeepFraction(e.target.value)}
                  className="w-24 rounded-md border border-gray-800 bg-gray-950 px-2 py-1 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </label>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-300">Loaders</h2>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={loadArchitecture}
              disabled={loadingArchitecture}
              className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-300 transition-colors hover:border-gray-500 hover:text-white disabled:opacity-50"
            >
              {loadingArchitecture ? "Loading Architecture..." : "Load Architecture"}
            </button>
            <button
              onClick={loadCheckpoints}
              disabled={loadingCheckpoints}
              className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-300 transition-colors hover:border-gray-500 hover:text-white disabled:opacity-50"
            >
              {loadingCheckpoints ? "Loading Checkpoints..." : "Refresh Checkpoints"}
            </button>
          </div>
        </section>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-300">Architecture Summary</h2>
          </div>
          {architecture ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <StatCard label="Model" value={architecture.model_name || "Unknown"} />
                <StatCard label="d_model" value={String(architecture.d_model)} />
                <StatCard label="MoE Layers" value={String(architecture.moe_layers.length)} />
                <StatCard label="DeltaNet Layers" value={String(architecture.deltanet_layers.length)} />
              </div>
              <pre className="overflow-x-auto rounded-lg border border-gray-800 bg-gray-950 p-3 text-xs text-gray-300">
                {architecture.summary}
              </pre>
            </div>
          ) : (
            <p className="text-sm text-gray-500">Load architecture to inspect the model layout.</p>
          )}
        </section>

        <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-300">Available Checkpoints</h2>
          </div>
          {checkpoints.length > 0 ? (
            <div className="space-y-2">
              {checkpoints.map((checkpoint) => (
                <div
                  key={checkpoint.path}
                  className="rounded-lg border border-gray-800 bg-gray-950 px-3 py-2"
                >
                  <p className="text-sm text-white">{checkpoint.name}</p>
                  <p className="mt-1 text-xs text-gray-500">{checkpoint.path}</p>
                  <p className="mt-1 text-xs text-gray-400">{checkpoint.size_mb} MB</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No saved SAE or CLT checkpoints found.</p>
          )}
        </section>
      </div>

      <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-300">Trace Result</h2>
        </div>
        {traceResult ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
              <StatCard label="Target Token" value={traceResult.target_token || "—"} />
              <StatCard label="Target Position" value={String(traceResult.target_position)} />
              <StatCard label="Nodes" value={String(traceResult.num_nodes)} />
              <StatCard label="Edges" value={String(traceResult.num_edges)} />
              <StatCard label="Attribution" value={traceResult.total_attribution.toFixed(4)} />
            </div>

            {traceResult.graph_json_path ? (
              <div className="rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-xs text-gray-400">
                Graph JSON saved to {traceResult.graph_json_path}
              </div>
            ) : null}

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Top Features
                </h3>
                <div className="space-y-2">
                  {traceResult.top_features.slice(0, 10).map((feature) => (
                    <div
                      key={`${feature.layer}-${feature.position}-${feature.feature_idx}`}
                      className="rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span>
                          L{feature.layer} · P{feature.position} · F{feature.feature_idx}
                        </span>
                        <span className="font-mono text-xs text-blue-300">
                          {feature.activation.toFixed(4)}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-gray-500">Token: {feature.token || "—"}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Top Edges
                </h3>
                <div className="space-y-2">
                  {traceResult.top_edges.slice(0, 10).map((edge, index) => (
                    <div
                      key={`${edge.source}-${edge.target}-${index}`}
                      className="rounded-lg border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300"
                    >
                      <p className="truncate text-xs text-gray-400">{edge.source}</p>
                      <p className="truncate text-xs text-gray-500">{edge.target}</p>
                      <p className="mt-1 font-mono text-xs text-amber-300">{edge.weight.toFixed(4)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500">Run a trace to inspect the active circuit.</p>
        )}
      </section>
    </div>
  );
}