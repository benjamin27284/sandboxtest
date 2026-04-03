"use client";

/**
 * Control Panel — Exogenous Shock Injector (the "do-operator").
 *
 * Allows the user to inject macroeconomic events into the live simulation
 * by POSTing to the API gateway.
 */

import { useState } from "react";
import type { ExogenousShock } from "@/types/market";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const PRESET_SHOCKS = [
  {
    label: "Rate Hike (+150bp)",
    category: "monetary_policy",
    headline: "Central bank unexpectedly raises interest rates by 150 basis points",
    body: "",
    severity: 0.9,
  },
  {
    label: "Earnings Miss",
    category: "earnings",
    headline: "Major tech company reports significant earnings miss; guidance lowered",
    body: "",
    severity: 0.7,
  },
  {
    label: "Geopolitical Crisis",
    category: "geopolitical",
    headline: "Escalating geopolitical tensions disrupt global supply chains",
    body: "",
    severity: 0.8,
  },
  {
    label: "Dovish Pivot",
    category: "monetary_policy",
    headline: "Fed signals pause in rate hikes; markets anticipate easing cycle",
    body: "",
    severity: 0.6,
  },
];

interface ShockInjectorProps {
  currentTick: number;
  onShockInjected?: () => void;
}

export function ShockInjector({ currentTick, onShockInjected }: ShockInjectorProps) {
  const [status, setStatus] = useState<string | null>(null);
  const [customHeadline, setCustomHeadline] = useState("");
  const [customSeverity, setCustomSeverity] = useState(0.5);

  const injectShock = async (shock: Omit<ExogenousShock, "tick">) => {
    setStatus("Injecting...");
    try {
      const res = await fetch(`${API_URL}/api/shocks/inject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...shock, tick: currentTick }),
      });
      const data = await res.json();
      setStatus(`Injected: ${data.shock_id}`);
      onShockInjected?.();
    } catch (err) {
      setStatus("Error: API unreachable");
    }
    setTimeout(() => setStatus(null), 3000);
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 space-y-4">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
        Event Injection (do-operator)
      </h3>

      {/* Preset shocks */}
      <div className="grid grid-cols-2 gap-2">
        {PRESET_SHOCKS.map((preset) => (
          <button
            key={preset.label}
            onClick={() => injectShock(preset)}
            className="px-3 py-2 text-xs bg-gray-800 hover:bg-red-900/50 border border-gray-600
                       hover:border-red-500 rounded text-gray-300 hover:text-red-300 transition-colors"
          >
            {preset.label}
            <span className="block text-[10px] text-gray-500 mt-0.5">
              severity: {preset.severity}
            </span>
          </button>
        ))}
      </div>

      {/* Custom shock */}
      <div className="space-y-2 border-t border-gray-700 pt-3">
        <input
          type="text"
          placeholder="Custom news headline..."
          value={customHeadline}
          onChange={(e) => setCustomHeadline(e.target.value)}
          className="w-full px-3 py-1.5 text-xs bg-gray-800 border border-gray-600 rounded
                     text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
        />
        <div className="flex items-center gap-3">
          <label className="text-xs text-gray-400">Severity:</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={customSeverity}
            onChange={(e) => setCustomSeverity(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="text-xs text-gray-400 w-8">{customSeverity}</span>
          <button
            onClick={() =>
              customHeadline &&
              injectShock({
                category: "custom",
                headline: customHeadline,
                body: "",
                severity: customSeverity,
              })
            }
            disabled={!customHeadline}
            className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-500 rounded text-white
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Inject
          </button>
        </div>
      </div>

      {/* Status */}
      {status && (
        <p className="text-xs text-amber-400 animate-pulse">{status}</p>
      )}
    </div>
  );
}
