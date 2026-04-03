// Shared TypeScript types for market data consumed via WebSocket.

export interface PriceLevel {
  price: number;
  quantity: number;
  count: number;
}

export interface OrderBookSnapshot {
  tick: number;
  mid_price: number;
  best_bid: number;
  best_ask: number;
  spread: number;
  bids: PriceLevel[];
  asks: PriceLevel[];
}

export interface TickSummary {
  tick: number;
  open_price: number;
  close_price: number;
  high_price: number;
  low_price: number;
  volume: number;
  num_trades: number;
  book_snapshot?: OrderBookSnapshot;
  consensus: number;
  disagreement: number;
  mass_signal: number;
}

export interface Execution {
  exec_id: string;
  order_id: string;
  agent_id: string;
  fill_price: number;
  fill_qty: number;
  timestamp: number;
  counter_order_id: string;
  counter_agent_id: string;
  aggressor_side: number;
  // Derived fields from proto_codec.decode_execution()
  buyer_id: string;
  seller_id: string;
  price: number;
  quantity: number;
}

export interface ExogenousShock {
  tick: number;
  category: string;
  headline: string;
  body: string;
  severity: number;
}

export interface EGCIRFReport {
  intervention: string;
  target: string;
  n_runs: number;
  mean_response: number[];
  std_response: number[];
  peak_effect: number;
  peak_tick: number;
  shock_tick: number;
  generated_at: number;
}

// Per-asset agent-driven simulation result (from event impact analysis)
export interface AssetSimulationResult {
  asset: string;
  ticker: string | null;
  direction: "up" | "down";
  magnitude: "high" | "medium" | "low";
  confidence: number;
  reason: string;
  simulation_method: "agent_lob";
  n_agents: number;
  n_ticks: number;
  base_price: number;
  price_trajectory: number[];
  ohlcv: {
    tick: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    trades: number;
  }[];
  total_volume: number;
  total_trades: number;
  peak_effect_pct: number;
  peak_tick: number;
  final_price: number;
  final_effect_pct: number;
}

export interface EventAnalysis {
  event: string;
  event_category: string;
  severity: number;
  time_horizon: string;
  impacts: {
    asset: string;
    ticker: string | null;
    direction: "up" | "down";
    magnitude: "high" | "medium" | "low";
    confidence: number;
    reason: string;
  }[];
  simulation_results: AssetSimulationResult[];
}

export type WSMessage = {
  topic: "market_data" | "tick_summary" | "executions";
  data: OrderBookSnapshot | TickSummary | Execution;
  timestamp: number;
};
