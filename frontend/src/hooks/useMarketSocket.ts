"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { WSMessage, TickSummary, OrderBookSnapshot } from "@/types/market";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/market";
const MAX_PRICE_HISTORY = 500;

interface MarketState {
  connected: boolean;
  priceHistory: { tick: number; price: number; volume: number }[];
  currentBook: OrderBookSnapshot | null;
  latestTick: TickSummary | null;
}

export function useMarketSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<MarketState>({
    connected: false,
    priceHistory: [],
    currentBook: null,
    latestTick: null,
  });

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((prev) => ({ ...prev, connected: true }));
    };

    ws.onclose = () => {
      setState((prev) => ({ ...prev, connected: false }));
      // Auto-reconnect after 2 seconds
      setTimeout(connect, 2000);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        setState((prev) => {
          const next = { ...prev };

          if (msg.topic === "tick_summary") {
            const tick = msg.data as TickSummary;
            next.latestTick = tick;
            next.priceHistory = [
              ...prev.priceHistory.slice(-(MAX_PRICE_HISTORY - 1)),
              { tick: tick.tick, price: tick.close_price, volume: tick.volume },
            ];
            if (tick.book_snapshot) {
              next.currentBook = tick.book_snapshot;
            }
          } else if (msg.topic === "market_data") {
            next.currentBook = msg.data as OrderBookSnapshot;
          }

          return next;
        });
      } catch {
        // ignore malformed messages
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return state;
}
