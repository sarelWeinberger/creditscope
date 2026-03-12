import React, { useState } from "react";
import { CoTConfig, ThinkingMode, ThinkingVisibility } from "../types";

interface ThinkingBudgetSliderProps {
  config: CoTConfig;
  onChange: (config: CoTConfig) => void;
}

const PRESETS = [
  { name: "none",     label: "None",     tokens: 0,    latency: "~0.1s" },
  { name: "minimal",  label: "Minimal",  tokens: 256,  latency: "~1s" },
  { name: "short",    label: "Short",    tokens: 512,  latency: "~2s" },
  { name: "standard", label: "Standard", tokens: 1024, latency: "~4s" },
  { name: "extended", label: "Extended", tokens: 2048, latency: "~8s" },
  { name: "deep",     label: "Deep",     tokens: 4096, latency: "~16s" },
  { name: "unlimited",label: "Unlimited",tokens: -1,   latency: "30s+" },
];

const QUICK_PRESETS = [
  { label: "Quick Lookup",      budget: "none",     mode: ThinkingMode.Off,  vis: ThinkingVisibility.Hidden },
  { label: "Standard Analysis", budget: "standard", mode: ThinkingMode.On,   vis: ThinkingVisibility.Collapsed },
  { label: "Deep Review",       budget: "deep",     mode: ThinkingMode.On,   vis: ThinkingVisibility.Streaming },
  { label: "Debug Mode",        budget: "unlimited",mode: ThinkingMode.On,   vis: ThinkingVisibility.Full },
];

const VISIBILITY_OPTIONS = [
  { value: ThinkingVisibility.Hidden,    label: "Hidden" },
  { value: ThinkingVisibility.Collapsed, label: "Collapsed" },
  { value: ThinkingVisibility.Streaming, label: "Streaming" },
  { value: ThinkingVisibility.Full,      label: "Full" },
];

export default function ThinkingBudgetSlider({ config, onChange }: ThinkingBudgetSliderProps) {
  const [expanded, setExpanded] = useState(false);
  const currentPresetIdx = PRESETS.findIndex((p) => p.name === config.budget);

  const thinkingOn = config.mode !== ThinkingMode.Off && config.budget !== "none";

  const applyPreset = (presetName: string) => {
    const preset = PRESETS.find((p) => p.name === presetName);
    if (!preset) return;
    onChange({
      ...config,
      budget: presetName,
      mode: presetName === "none" ? ThinkingMode.Off : ThinkingMode.On,
      enable_thinking: presetName !== "none",
      visibility:
        presetName === "none"
          ? ThinkingVisibility.Hidden
          : presetName === "deep" || presetName === "unlimited"
          ? ThinkingVisibility.Streaming
          : ThinkingVisibility.Collapsed,
    });
  };

  const currentPreset = PRESETS[currentPresetIdx] || PRESETS[3];

  return (
    <div className="select-none">
      {/* Compact row */}
      <div className="flex items-center gap-3">
        {/* Toggle */}
        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              onChange({
                ...config,
                mode: thinkingOn ? ThinkingMode.Off : ThinkingMode.On,
                budget: thinkingOn ? "none" : "unlimited",
                enable_thinking: !thinkingOn,
              })
            }
            className={`relative w-9 h-5 rounded-full transition-colors ${
              thinkingOn ? "bg-purple-600" : "bg-gray-700"
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                thinkingOn ? "translate-x-4" : ""
              }`}
            />
          </button>
          <span className="text-xs text-gray-400">
            {thinkingOn ? (
              <span className="text-purple-400">
                Thinking: <strong>{currentPreset.label}</strong>
              </span>
            ) : (
              "No thinking"
            )}
          </span>
        </div>

        {/* Slider */}
        {thinkingOn && (
          <input
            type="range"
            min={1}
            max={PRESETS.length - 1}
            value={Math.max(1, currentPresetIdx)}
            onChange={(e) => applyPreset(PRESETS[parseInt(e.target.value)].name)}
            className="flex-1 h-1 accent-purple-500 cursor-pointer"
          />
        )}

        {/* Latency indicator */}
        {thinkingOn && (
          <span className="text-xs text-gray-500 w-12 text-right">
            {currentPreset.latency}
          </span>
        )}

        {/* Expand */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-gray-500 hover:text-gray-300 transition-colors ml-1"
        >
          <svg
            className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Expanded section */}
      {expanded && (
        <div className="mt-3 bg-gray-800 rounded-xl p-3 space-y-3">
          {/* Quick presets */}
          <div>
            <p className="text-xs text-gray-500 mb-2">Quick Presets</p>
            <div className="grid grid-cols-2 gap-1.5">
              {QUICK_PRESETS.map((qp) => (
                <button
                  key={qp.label}
                  onClick={() =>
                    onChange({
                      ...config,
                      budget: qp.budget,
                      mode: qp.mode,
                      visibility: qp.vis,
                      enable_thinking: qp.mode !== ThinkingMode.Off,
                    })
                  }
                  className={`px-2 py-1.5 text-xs rounded-lg border text-left transition-colors ${
                    config.budget === qp.budget
                      ? "border-purple-500 bg-purple-950 text-purple-300"
                      : "border-gray-700 text-gray-400 hover:text-white hover:border-gray-600"
                  }`}
                >
                  {qp.label}
                </button>
              ))}
            </div>
          </div>

          {/* All preset stops */}
          <div>
            <p className="text-xs text-gray-500 mb-2">Budget</p>
            <div className="flex items-center gap-1">
              {PRESETS.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => applyPreset(preset.name)}
                  className={`flex-1 py-1 text-xs rounded transition-colors ${
                    config.budget === preset.name
                      ? "bg-purple-600 text-white"
                      : "bg-gray-700 text-gray-400 hover:bg-gray-600"
                  }`}
                  title={`${preset.tokens === -1 ? "∞" : preset.tokens} tokens · ${preset.latency}`}
                >
                  {preset.label.slice(0, 3)}
                </button>
              ))}
            </div>
          </div>

          {/* Visibility */}
          <div>
            <p className="text-xs text-gray-500 mb-2">Visibility</p>
            <div className="flex items-center gap-1">
              {VISIBILITY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => onChange({ ...config, visibility: opt.value })}
                  className={`flex-1 py-1 text-xs rounded transition-colors ${
                    config.visibility === opt.value
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 text-gray-400 hover:bg-gray-600"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Auto mode toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-300">Auto-classify complexity</p>
              <p className="text-xs text-gray-500">Automatically select budget based on query</p>
            </div>
            <button
              onClick={() =>
                onChange({
                  ...config,
                  mode: config.mode === ThinkingMode.Auto ? ThinkingMode.On : ThinkingMode.Auto,
                  auto_classify: config.mode !== ThinkingMode.Auto,
                })
              }
              className={`relative w-9 h-5 rounded-full transition-colors ${
                config.mode === ThinkingMode.Auto ? "bg-blue-600" : "bg-gray-700"
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                  config.mode === ThinkingMode.Auto ? "translate-x-4" : ""
                }`}
              />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
