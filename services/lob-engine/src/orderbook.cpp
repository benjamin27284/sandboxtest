// ============================================================================
// OrderBook — implementation of the Continuous Double Auction LOB.
//
// Matching semantics:
//   • add_order()    — O(log P) insert into the correct side.
//   • match_orders() — O(F · log P) sweep, F = number of fills produced.
//     Walks best bid vs best ask while they cross, FIFO within each level.
//     Fill price = resting (passive) order's price (price improvement for the
//     aggressor), determined by which order has the lower sequence number.
//
// Zero-allocation focus:
//   • Orders stored by value in std::deque (cache-friendly, no shared_ptr).
//   • Execution vector reserved to 64 to avoid early reallocs in hot path.
//   • Stale orders (filled/cancelled) drained lazily from level front.
//   • make_exec_id uses std::format with a monotonic counter — single alloc.
// ============================================================================

#include "orderbook.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <utility>

namespace abms::lob {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double kInf = std::numeric_limits<double>::infinity();

static std::int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ─── add_order ──────────────────────────────────────────────────────────────

void OrderBook::add_order(Order order) {
    order.sequence  = seq_counter_++;
    order.remaining = order.quantity;

    // Market orders get a synthetic price that guarantees crossing.
    if (order.type == OrderType::Market) {
        order.price = (order.side == Side::Buy) ? kInf : 0.0;
    }

    // Index for O(1) cancel.
    order_index_[order.order_id] = OrderLoc{order.side, order.price};

    if (order.side == Side::Buy) {
        bids_[order.price].push_back(std::move(order));
    } else {
        asks_[order.price].push_back(std::move(order));
    }
}

// ─── match_orders ───────────────────────────────────────────────────────────

std::vector<Execution> OrderBook::match_orders() {
    std::vector<Execution> fills;
    fills.reserve(64);  // avoid early reallocs in the common case

    while (!bids_.empty() && !asks_.empty()) {
        auto bid_it = bids_.begin();
        auto ask_it = asks_.begin();

        // No crossing — done.
        if (bid_it->first < ask_it->first) break;

        auto& bid_deque = bid_it->second;
        auto& ask_deque = ask_it->second;

        // Drain any stale (filled/cancelled) orders from the front.
        while (!bid_deque.empty() && bid_deque.front().is_filled())
            bid_deque.pop_front();
        while (!ask_deque.empty() && ask_deque.front().is_filled())
            ask_deque.pop_front();

        if (bid_deque.empty()) { bids_.erase(bid_it); continue; }
        if (ask_deque.empty()) { asks_.erase(ask_it); continue; }

        Order& bid = bid_deque.front();
        Order& ask = ask_deque.front();

        // Fill quantity: min of both remaining.
        const std::int32_t fill_qty = std::min(bid.remaining, ask.remaining);

        // Fill price: passive (resting) order's price.
        // The order with the *lower* sequence number arrived first → passive.
        const double fill_price = (bid.sequence < ask.sequence)
                                      ? bid.price    // bid was resting
                                      : ask.price;   // ask was resting

        // Determine aggressor for the Execution report.
        const bool bid_is_aggressor = (bid.sequence > ask.sequence);
        const Order& aggressor = bid_is_aggressor ? bid : ask;
        const Order& passive   = bid_is_aggressor ? ask : bid;

        fills.push_back(Execution{
            .exec_id          = make_exec_id(exec_counter_++),
            .order_id         = aggressor.order_id,
            .agent_id         = aggressor.agent_id,
            .fill_price       = fill_price,
            .fill_qty         = fill_qty,
            .timestamp        = now_ns(),
            .counter_order_id = passive.order_id,
            .counter_agent_id = passive.agent_id,
            .aggressor_side   = aggressor.side,
        });

        // Decrement remaining quantities.
        bid.remaining -= fill_qty;
        ask.remaining -= fill_qty;

        // Pop fully filled orders and clean empty levels.
        if (bid.is_filled()) {
            order_index_.erase(bid.order_id);
            bid_deque.pop_front();
        }
        if (ask.is_filled()) {
            order_index_.erase(ask.order_id);
            ask_deque.pop_front();
        }
        if (bid_deque.empty()) bids_.erase(bid_it);
        if (ask_deque.empty()) asks_.erase(ask_it);
    }

    return fills;
}

// ─── cancel ─────────────────────────────────────────────────────────────────

bool OrderBook::cancel(const std::string& order_id) {
    auto idx_it = order_index_.find(order_id);
    if (idx_it == order_index_.end()) return false;

    const auto [side, price] = idx_it->second;

    // Lambda: find and tombstone the order within the level's deque.
    auto tombstone = [&](auto& book_side) -> bool {
        auto level_it = book_side.find(price);
        if (level_it == book_side.end()) return false;

        auto& deq = level_it->second;
        for (auto& o : deq) {
            if (o.order_id == order_id) {
                o.remaining = 0;  // tombstone; drained lazily in match_orders
                order_index_.erase(idx_it);
                return true;
            }
        }
        return false;
    };

    return (side == Side::Buy) ? tombstone(bids_) : tombstone(asks_);
}

// ─── Accessors ──────────────────────────────────────────────────────────────

double OrderBook::best_bid_price() const noexcept {
    return bids_.empty() ? kNaN : bids_.begin()->first;
}

double OrderBook::best_ask_price() const noexcept {
    return asks_.empty() ? kNaN : asks_.begin()->first;
}

double OrderBook::get_mid_price() const noexcept {
    const double bb = best_bid_price();
    const double ba = best_ask_price();
    if (std::isnan(bb) || std::isnan(ba)) return kNaN;
    return (bb + ba) * 0.5;
}

double OrderBook::spread() const noexcept {
    const double bb = best_bid_price();
    const double ba = best_ask_price();
    if (std::isnan(bb) || std::isnan(ba)) return kNaN;
    return ba - bb;
}

std::int32_t OrderBook::bid_depth() const noexcept {
    std::int32_t sum = 0;
    for (const auto& [_, deq] : bids_)
        for (const auto& o : deq) sum += o.remaining;
    return sum;
}

std::int32_t OrderBook::ask_depth() const noexcept {
    std::int32_t sum = 0;
    for (const auto& [_, deq] : asks_)
        for (const auto& o : deq) sum += o.remaining;
    return sum;
}

std::int32_t OrderBook::total_volume() const noexcept {
    return bid_depth() + ask_depth();
}

// ─── Snapshot ───────────────────────────────────────────────────────────────

BookSnapshot OrderBook::snapshot(int depth) const {
    BookSnapshot snap{
        .mid_price    = get_mid_price(),
        .best_bid     = best_bid_price(),
        .best_ask     = best_ask_price(),
        .spread       = spread(),
        .total_volume = total_volume(),
    };

    int count = 0;
    for (const auto& [price, deq] : bids_) {
        if (count++ >= depth) break;
        std::int32_t qty = 0;
        for (const auto& o : deq) qty += o.remaining;
        snap.bid_levels.push_back({price, qty, static_cast<std::int32_t>(deq.size())});
    }
    count = 0;
    for (const auto& [price, deq] : asks_) {
        if (count++ >= depth) break;
        std::int32_t qty = 0;
        for (const auto& o : deq) qty += o.remaining;
        snap.ask_levels.push_back({price, qty, static_cast<std::int32_t>(deq.size())});
    }

    return snap;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

std::string OrderBook::make_exec_id(std::uint64_t counter) {
    return std::format("EX-{:012d}", counter);
}

}  // namespace abms::lob
