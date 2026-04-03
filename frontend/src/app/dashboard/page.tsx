"use client";

/**
 * Dashboard — main simulation monitoring page.
 *
 * Connects to the API gateway via WebSocket and displays:
 *   - Real-time price chart with event markers
 *   - Multi-asset impact simulation chart
 *   - Shock injection control panel
 *   - Live telemetry stats
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { useMarketSocket } from "@/hooks/useMarketSocket";
import { PriceChart } from "@/components/charts/PriceChart";
import { MultiAssetChart } from "@/components/charts/MultiAssetChart";
import { ShockInjector } from "@/components/controls/ShockInjector";
import type { EventAnalysis } from "@/types/market";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function DashboardPage() {
  const { connected, priceHistory, currentBook, latestTick } = useMarketSocket();
  const [simRunning, setSimRunning] = useState(true);
  const [eventAnalysis, setEventAnalysis] = useState<EventAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentTick = latestTick?.tick ?? 0;

  const controlSim = useCallback(async (action: "stop" | "start") => {
    try {
      const res = await fetch(`${API_URL}/api/simulation/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      if (res.ok) {
        setSimRunning(action === "start");
      }
    } catch {
      // API unreachable
    }
  }, []);

  const fetchEventAnalysis = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/event-analysis/latest`);
      const data = await res.json();
      if (data.analysis && data.analysis.simulation_results?.length > 0) {
        setEventAnalysis(data.analysis);
        setAnalysisLoading(false);
        // Stop polling
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      }
    } catch {
      // API unreachable
    }
  }, []);

  const onShockInjected = useCallback(() => {
    setAnalysisLoading(true);
    // Start polling for analysis results (agent-driven simulation takes time)
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(fetchEventAnalysis, 3000);
    // Also fetch once on page load in case there's existing data
    fetchEventAnalysis();
  }, [fetchEventAnalysis]);

  // Fetch existing analysis on mount
  useEffect(() => {
    fetchEventAnalysis();
  }, [fetchEventAnalysis]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold">ABMS Market Simulation</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            1,000 LLM-powered agents / Continuous Double Auction
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-400">Tick: {currentTick}</span>
          <span className={`text-xs flex items-center gap-1 ${connected ? "text-green-400" : "text-red-400"}`}>
            <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400" : "bg-red-400"}`} />
            {connected ? "Live" : "Disconnected"}
          </span>
          {simRunning ? (
            <button
              onClick={() => controlSim("stop")}
              className="px-3 py-1 text-xs bg-red-600 hover:bg-red-500 rounded text-white transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={() => controlSim("start")}
              className="px-3 py-1 text-xs bg-green-600 hover:bg-green-500 rounded text-white transition-colors"
            >
              Start
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* ── Charts (8 cols) ──────────────────────────────────────── */}
        <div className="col-span-8 space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-gray-400 mb-2">Price Time-Series</h2>
            <PriceChart data={priceHistory} shockTicks={[25, 50, 75]} />
          </div>

          {/* Multi-Asset Impact Chart */}
          {analysisLoading && !eventAnalysis && (
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-8 text-center">
              <p className="text-gray-500 text-sm animate-pulse">
                Running agent-driven multi-asset simulation...
              </p>
            </div>
          )}
          {eventAnalysis && eventAnalysis.simulation_results?.length > 0 && (
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-sm font-semibold text-gray-400">
                  Event: {eventAnalysis.event_category}
                </h2>
                <span className="text-xs text-gray-500">
                  {eventAnalysis.simulation_results.length} assets simulated
                </span>
              </div>
              <MultiAssetChart results={eventAnalysis.simulation_results} />
            </div>
          )}

          {/* Telemetry bar */}
          <div className="grid grid-cols-5 gap-3">
            {[
              { label: "Mid Price", value: currentBook?.mid_price?.toFixed(2) ?? (latestTick ? latestTick.close_price.toFixed(2) : "\u2014") },
              { label: "Spread", value: currentBook?.spread?.toFixed(4) ?? (latestTick ? (latestTick.high_price - latestTick.low_price).toFixed(4) : "\u2014") },
              { label: "Consensus", value: latestTick != null && latestTick.consensus != null ? latestTick.consensus.toFixed(3) : "\u2014" },
              { label: "MASS Signal", value: latestTick != null && latestTick.mass_signal != null ? latestTick.mass_signal.toFixed(3) : "\u2014" },
              { label: "Volume", value: latestTick != null && latestTick.volume != null ? String(latestTick.volume) : "\u2014" },
            ].map((stat) => (
              <div key={stat.label} className="bg-gray-900 rounded border border-gray-700 p-3">
                <p className="text-[10px] text-gray-500 uppercase">{stat.label}</p>
                <p className="text-lg font-mono font-bold">{stat.value}</p>
              </div>
            ))}
          </div>

          {/* Order book depth */}
          {currentBook && (
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded border border-gray-700 p-3">
                <h3 className="text-xs text-green-400 font-semibold mb-2">Bids</h3>
                <div className="space-y-0.5 font-mono text-xs">
                  {(currentBook.bids ?? []).slice(0, 8).map((lvl, i) => (
                    <div key={i} className="flex justify-between text-green-300/80">
                      <span>{lvl.price.toFixed(2)}</span>
                      <span>{lvl.quantity}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-gray-900 rounded border border-gray-700 p-3">
                <h3 className="text-xs text-red-400 font-semibold mb-2">Asks</h3>
                <div className="space-y-0.5 font-mono text-xs">
                  {(currentBook.asks ?? []).slice(0, 8).map((lvl, i) => (
                    <div key={i} className="flex justify-between text-red-300/80">
                      <span>{lvl.price.toFixed(2)}</span>
                      <span>{lvl.quantity}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Control Panel (4 cols) ────────────────────────────────────── */}
        <div className="col-span-4 space-y-4">
          <ShockInjector currentTick={currentTick} onShockInjected={onShockInjected} />

          {/* EGCIRF Export */}
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
              Causal Inference Reports
            </h3>
            <button
              onClick={async () => {
                const res = await fetch(
                  `${API_URL}/api/reports/egcirf?shock_tick=${currentTick}`
                );
                const data = await res.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `egcirf_report_tick${data.shock_tick ?? currentTick}.json`;
                a.click();
              }}
              className="w-full px-3 py-2 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-600
                         rounded text-gray-300 transition-colors"
            >
              Export EGCIRF Report (JSON)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
