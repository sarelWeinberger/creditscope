import { useCallback, useEffect, useRef, useState } from "react";
import { EntropyDataPoint, ExpertHeatmap, LayerActivity } from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

interface ObservabilityState {
  heatmapData: ExpertHeatmap | null;
  entropyData: EntropyDataPoint[];
  layerData: LayerActivity[];
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  autoRefresh: boolean;
  setAutoRefresh: (v: boolean) => void;
  refresh: () => void;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export function useObservability(): ObservabilityState {
  const [heatmapData, setHeatmapData] = useState<ExpertHeatmap | null>(null);
  const [entropyData, setEntropyData] = useState<EntropyDataPoint[]>([]);
  const [layerData, setLayerData] = useState<LayerActivity[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const heatmapTimer = useRef<number | null>(null);
  const entropyTimer = useRef<number | null>(null);
  const layerTimer = useRef<number | null>(null);

  const fetchHeatmap = useCallback(async () => {
    try {
      const data = await fetchJson<{ heatmap: ExpertHeatmap }>(
        `${API_BASE}/observability/moe/heatmap`
      );
      if (data.heatmap) {
        setHeatmapData(data.heatmap);
      }
    } catch (e) {
      // Silently ignore connectivity errors during polling
    }
  }, []);

  const fetchEntropy = useCallback(async () => {
    try {
      const data = await fetchJson<{ data: EntropyDataPoint[] }>(
        `${API_BASE}/observability/moe/entropy?window=200`
      );
      if (data.data) {
        setEntropyData(data.data.slice(-200));
      }
    } catch (e) {
      // Silently ignore
    }
  }, []);

  const fetchLayers = useCallback(async () => {
    try {
      const data = await fetchJson<{ layers: LayerActivity[] }>(
        `${API_BASE}/observability/layers?top_k=8`
      );
      if (data.layers) {
        setLayerData(data.layers);
      }
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      await Promise.all([fetchHeatmap(), fetchEntropy(), fetchLayers()]);
    } finally {
      setIsLoading(false);
    }
  }, [fetchHeatmap, fetchEntropy, fetchLayers]);

  // Start / stop polling based on autoRefresh
  useEffect(() => {
    if (!autoRefresh) {
      if (heatmapTimer.current !== null) clearInterval(heatmapTimer.current);
      if (entropyTimer.current !== null) clearInterval(entropyTimer.current);
      if (layerTimer.current !== null) clearInterval(layerTimer.current);
      heatmapTimer.current = null;
      entropyTimer.current = null;
      layerTimer.current = null;
      return;
    }

    // Initial fetch
    refresh();

    // Heatmap: every 15s
    heatmapTimer.current = window.setInterval(fetchHeatmap, 15000);
    // Entropy: every 10s
    entropyTimer.current = window.setInterval(fetchEntropy, 10000);
    // Layers: every 15s
    layerTimer.current = window.setInterval(fetchLayers, 15000);

    return () => {
      if (heatmapTimer.current !== null) clearInterval(heatmapTimer.current);
      if (entropyTimer.current !== null) clearInterval(entropyTimer.current);
      if (layerTimer.current !== null) clearInterval(layerTimer.current);
    };
  }, [autoRefresh, refresh, fetchHeatmap, fetchEntropy, fetchLayers]);

  return {
    heatmapData,
    entropyData,
    layerData,
    isLoading,
    error,
    lastUpdated,
    autoRefresh,
    setAutoRefresh,
    refresh,
  };
}
