#pragma once
// ============================================================================
// OrderBook — ultra-low-latency Continuous Double Auction LOB.
//
// Data structures
// ───────────────
//   Bids: std::map<double, std::deque<Order>, std::greater<double>>
//         → highest price first, O(log P) insert, O(1) best-price peek
//   Asks: std::map<double, std::deque<Order>, std::less<double>>
//         → lowest price first, same complexities
//
// Matching: price-time FIFO.  match_orders() sweeps while best_bid ≥ best_ask.
//
// Zero-allocation policy
// ──────────────────────
//   • Orders stored by value in deque (no shared_ptr, no indirection).
//   • Execution vector pre-reserved to amortise heap traffic.
//   • Cancel via lazy tombstone (remaining = 0) cleaned during match.
//   • order_index_ gives O(1) locate for explicit cancel.
//
// Target: ≥854 K orders/sec on a single core (release build, -O3 -march=native).
// ============================================================================

#include "order.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace abms::lob {

// ─── Lightweight snapshot for downstream consumers ──────────────────────────

struct BookSnapshot {
    double   mid_price;
    double   best_bid;
    double   best_ask;
    double   spread;
    std::int32_t total_volume;

    struct Level { double price; std::int32_t qty; std::int32_t count; };
    std::vector<Level> bid_levels;
    std::vector<Level> ask_levels;
};

// ─── OrderBook ──────────────────────────────────────────────────────────────

class OrderBook {
public:
    OrderBook() = default;

    // ── Core API (required by spec) ─────────────────────────────────────────

    /// Insert an order into the book on the appropriate side.
    /// For MARKET orders the price is synthetically set to guarantee crossing.
    void add_order(Order order);

    /// Sweep the book while best_bid ≥ best_ask, executing price-time FIFO
    /// matches.  Returns all fills produced in this sweep.
    [[nodiscard]] std::vector<Execution> match_orders();

    /// (best_bid + best_ask) / 2, or NaN if either side is empty.
    [[nodiscard]] double get_mid_price() const noexcept;

    // ── Cancel ──────────────────────────────────────────────────────────────

    /// Cancel a resting order by ID.  Returns true if found and removed.
    bool cancel(const std::string& order_id);

    // ── Accessors ───────────────────────────────────────────────────────────

    [[nodiscard]] double   best_bid_price()  const noexcept;
    [[nodiscard]] double   best_ask_price()  const noexcept;
    [[nodiscard]] double   spread()          const noexcept;
    [[nodiscard]] std::int32_t bid_depth()   const noexcept;
    [[nodiscard]] std::int32_t ask_depth()   const noexcept;
    [[nodiscard]] std::int32_t total_volume() const noexcept;

    [[nodiscard]] BookSnapshot snapshot(int depth = 10) const;

    // ── Statistics ──────────────────────────────────────────────────────────

    [[nodiscard]] std::uint64_t total_orders_processed() const noexcept { return seq_counter_; }
    [[nodiscard]] std::uint64_t total_executions()       const noexcept { return exec_counter_; }

private:
    // Bids: descending price (highest first)  — O(log P) insert
    std::map<double, std::deque<Order>, std::greater<double>> bids_;
    // Asks: ascending price  (lowest first)   — O(log P) insert
    std::map<double, std::deque<Order>, std::less<double>>    asks_;

    // O(1) cancel lookup: order_id → location in the book
    struct OrderLoc { Side side; double price; };
    std::unordered_map<std::string, OrderLoc> order_index_;

    std::uint64_t seq_counter_  = 0;
    std::uint64_t exec_counter_ = 0;

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Drain filled / cancelled orders from the front of a price level.
    /// Returns true if the level was fully emptied (caller should erase it).
    template <typename MapIt>
    static bool drain_stale(MapIt level_it);

    static std::string make_exec_id(std::uint64_t counter);
};

}  // namespace abms::lob
