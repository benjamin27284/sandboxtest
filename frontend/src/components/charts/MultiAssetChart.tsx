"use client";

import type { AssetSimulationResult } from "@/types/market";

const COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#a855f7",
  "#6366f1", "#10b981", "#e11d48", "#0ea5e9", "#84cc16",
];

interface MultiAssetChartProps {
  results: AssetSimulationResult[];
  width?: number;
  height?: number;
}

export function MultiAssetChart({
  results,
  width = 800,
  height = 350,
}: MultiAssetChartProps) {
  if (results.length === 0) return null;

  // Normalize to % change from base price
  const series = results.map((r) => ({
    asset: r.asset,
    direction: r.direction,
    finalPct: r.final_effect_pct,
    points: r.price_trajectory.map((p) => ((p - r.base_price) / r.base_price) * 100),
  }));

  const allPcts = series.flatMap((s) => s.points);
  const minPct = Math.min(...allPcts, 0);
  const maxPct = Math.max(...allPcts, 0);
  const range = maxPct - minPct || 1;
  const maxLen = Math.max(...series.map((s) => s.points.length));

  const padX = 50;
  const padY = 20;
  const padBottom = 30;
  const chartW = width - padX * 2;
  const chartH = height - padY - padBottom;

  const toX = (i: number) => padX + (i / Math.max(maxLen - 1, 1)) * chartW;
  const toY = (pct: number) => padY + chartH - ((pct - minPct) / range) * chartH;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
        Multi-Asset Impact Simulation
      </h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
          const y = padY + chartH * (1 - pct);
          const label = (minPct + range * pct).toFixed(1) + "%";
          return (
            <g key={pct}>
              <line x1={padX} y1={y} x2={width - padX} y2={y} stroke="#374151" strokeWidth={0.5} />
              <text x={padX - 4} y={y + 3} textAnchor="end" fill="#6b7280" fontSize={9}>
                {label}
              </text>
            </g>
          );
        })}

        {/* Zero line */}
        <line
          x1={padX} y1={toY(0)} x2={width - padX} y2={toY(0)}
          stroke="#6b7280" strokeWidth={1} strokeDasharray="4 2"
        />

        {/* Asset lines */}
        {series.map((s, si) => {
          const color = COLORS[si % COLORS.length];
          const path = s.points
            .map((p, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(p)}`)
            .join(" ");
          return (
            <g key={s.asset}>
              <path d={path} fill="none" stroke={color} strokeWidth={1.5} opacity={0.85} />
              {s.points.length > 0 && (
                <circle
                  cx={toX(s.points.length - 1)}
                  cy={toY(s.points[s.points.length - 1])}
                  r={3}
                  fill={color}
                />
              )}
            </g>
          );
        })}
      </svg>

      {/* Legend + results table */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        {series.map((s, si) => {
          const color = COLORS[si % COLORS.length];
          const r = results[si];
          return (
            <div key={s.asset} className="flex items-center gap-2 bg-gray-800 rounded px-2 py-1.5">
              <span className="w-3 h-0.5 flex-shrink-0" style={{ backgroundColor: color }} />
              <span className="text-gray-300 truncate">{s.asset}</span>
              <span className={`ml-auto font-mono ${s.finalPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                {s.finalPct >= 0 ? "+" : ""}{s.finalPct.toFixed(1)}%
              </span>
              <span className="text-gray-500">{r.total_trades}t</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
