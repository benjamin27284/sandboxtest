#pragma once
// ============================================================================
// Order / Execution — core value types for the LOB matching engine.
//
// Aligned to proto/market.proto (abms.market.Order, abms.market.Execution).
// All types are plain aggregates — no heap indirection, no shared_ptr.
// ============================================================================

#include <cstdint>
#include <string>

namespace abms::lob {

// ─── Enums (mirror proto Side / OrderType with explicit underlying) ─────────

enum class Side : std::uint8_t { Buy = 1, Sell = 2 };
enum class OrderType : std::uint8_t { Limit = 1, Market = 2 };

// ─── Order ──────────────────────────────────────────────────────────────────
//
// Stored **by value** inside std::deque<Order> at each price level.
// `remaining` is engine-internal bookkeeping, decremented on partial fills.
// `sequence`  is a monotonic counter assigned by OrderBook::add_order() and
//             used as the FIFO tie-breaker within a price level.

struct Order {
    std::string order_id;
    std::string agent_id;
    Side        side;
    OrderType   type;
    double      price;         // ignored for Market orders
    std::int32_t quantity;     // original quantity (proto: int32)
    std::int64_t timestamp;    // nanosecond epoch

    // ── Engine-internal state (not serialised to proto) ─────────────────────
    std::int32_t remaining = 0;   // decremented on partial fills
    std::uint64_t sequence = 0;   // monotonic insertion counter

    [[nodiscard]] bool is_filled() const noexcept { return remaining <= 0; }
};

// ─── Execution (fill report) ────────────────────────────────────────────────
//
// One Execution per matched fill.  Proto-facing fields come first; the two
// `counter_*` fields carry the passive side's identity for downstream
// settlement / PnL without requiring a second message per fill.

struct Execution {
    // Proto-aligned fields (abms.market.Execution)
    std::string  exec_id;
    std::string  order_id;         // aggressor order
    std::string  agent_id;         // aggressor agent
    double       fill_price;
    std::int32_t fill_qty;
    std::int64_t timestamp;

    // Engine-internal: passive (resting) counterparty
    std::string  counter_order_id;
    std::string  counter_agent_id;
    Side         aggressor_side;     // which side was the aggressor
};

}  // namespace abms::lob
