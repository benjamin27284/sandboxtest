"use client";

/**
 * Real-time price time-series chart using Lightweight Charts.
 *
 * Receives price history from the useMarketSocket hook and renders
 * a candlestick/line chart with event injection markers.
 *
 * In production: npm install lightweight-charts
 * For now: renders a simple SVG sparkline as a dependency-free fallback.
 */

import { useRef, useEffect } from "react";

interface PricePoint {
  tick: number;
  price: number;
  volume: number;
}

interface PriceChartProps {
  data: PricePoint[];
  shockTicks?: number[];
  width?: number;
  height?: number;
}

export function PriceChart({
  data,
  shockTicks = [],
  width = 800,
  height = 300,
}: PriceChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[300px] bg-gray-900 rounded-lg border border-gray-700">
        <p className="text-gray-500">Waiting for market data...</p>
      </div>
    );
  }

  const prices = data.map((d) => d.price);
  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);
  const range = maxP - minP || 1;

  const padX = 40;
  const padY = 20;
  const chartW = width - padX * 2;
  const chartH = height - padY * 2;

  const toX = (i: number) => padX + (i / Math.max(data.length - 1, 1)) * chartW;
  const toY = (p: number) => padY + chartH - ((p - minP) / range) * chartH;

  const linePath = data.map((d, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(d.price)}`).join(" ");

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
          const y = padY + chartH * (1 - pct);
          const label = (minP + range * pct).toFixed(2);
          return (
            <g key={pct}>
              <line x1={padX} y1={y} x2={width - padX} y2={y} stroke="#374151" strokeWidth={0.5} />
              <text x={padX - 4} y={y + 3} textAnchor="end" fill="#6b7280" fontSize={9}>
                {label}
              </text>
            </g>
          );
        })}

        {/* Shock markers */}
        {shockTicks.map((tick) => {
          const idx = data.findIndex((d) => d.tick === tick);
          if (idx < 0) return null;
          return (
            <line
              key={tick}
              x1={toX(idx)}
              y1={padY}
              x2={toX(idx)}
              y2={padY + chartH}
              stroke="#ef4444"
              strokeWidth={1.5}
              strokeDasharray="4 2"
              opacity={0.8}
            />
          );
        })}

        {/* Price line */}
        <path d={linePath} fill="none" stroke="#3b82f6" strokeWidth={1.5} />

        {/* Latest price dot */}
        {data.length > 0 && (
          <circle
            cx={toX(data.length - 1)}
            cy={toY(data[data.length - 1].price)}
            r={3}
            fill="#3b82f6"
          />
        )}
      </svg>
    </div>
  );
}
