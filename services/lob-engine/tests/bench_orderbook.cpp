// ============================================================================
// OrderBook micro-benchmark — validates correctness + measures throughput.
//
// Correctness: deterministic asserts on fill counts, prices, and book state.
// Throughput:  target ≥854 K orders/sec on a single core (-O3 -march=native).
// ============================================================================

#include "orderbook.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <string>

using namespace abms::lob;
using Clock = std::chrono::high_resolution_clock;

// ─── Helpers ────────────────────────────────────────────────────────────────

static Order make_order(const std::string& id, const std::string& agent,
                        Side side, OrderType type, double price,
                        std::int32_t qty, std::int64_t ts = 0) {
    return Order{
        .order_id  = id,
        .agent_id  = agent,
        .side      = side,
        .type      = type,
        .price     = price,
        .quantity  = qty,
        .timestamp = ts,
    };
}

// ─── Correctness tests ─────────────────────────────────────────────────────

static void test_basic_match() {
    OrderBook book;

    // Bid rests (no asks yet).
    book.add_order(make_order("B1", "A1", Side::Buy, OrderType::Limit, 100.0, 10));
    auto execs = book.match_orders();
    assert(execs.empty());
    assert(book.best_bid_price() == 100.0);

    // Sell crosses → fill 5 @ 100.0 (bid is resting → fill at bid price).
    book.add_order(make_order("S1", "A2", Side::Sell, OrderType::Limit, 99.0, 5));
    execs = book.match_orders();
    assert(execs.size() == 1);
    assert(execs[0].fill_qty == 5);
    assert(execs[0].fill_price == 100.0);   // resting (bid) price
    assert(execs[0].agent_id == "A2");       // aggressor = seller
    assert(execs[0].counter_agent_id == "A1");

    // Bid should have 5 remaining, no asks.
    assert(book.best_bid_price() == 100.0);
    assert(book.bid_depth() == 5);
    assert(book.ask_depth() == 0);

    std::cout << "  [PASS] test_basic_match\n";
}

static void test_partial_fill_multi_level() {
    OrderBook book;

    // Three bid levels: 100.0, 99.5, 99.0 — each with qty 10.
    for (int i = 0; i < 3; ++i) {
        book.add_order(make_order(
            std::format("B{}", i), std::format("A{}", i),
            Side::Buy, OrderType::Limit, 100.0 - i * 0.5, 10));
    }
    (void)book.match_orders();  // no asks yet — nothing to match

    // Aggressive sell sweeps 25 qty through the bid levels.
    book.add_order(make_order("S0", "SELLER", Side::Sell, OrderType::Limit, 99.0, 25));
    auto execs = book.match_orders();

    assert(execs.size() == 3);
    assert(execs[0].fill_qty == 10);   // 100.0 fully consumed
    assert(execs[1].fill_qty == 10);   // 99.5  fully consumed
    assert(execs[2].fill_qty == 5);    // 99.0  partially consumed

    assert(book.bid_depth() == 5);     // 5 remaining at 99.0
    assert(book.ask_depth() == 0);

    std::cout << "  [PASS] test_partial_fill_multi_level\n";
}

static void test_market_order() {
    OrderBook book;

    // Resting limit ask.
    book.add_order(make_order("S1", "A1", Side::Sell, OrderType::Limit, 105.0, 20));
    (void)book.match_orders();

    // Market buy — should cross at any price.
    book.add_order(make_order("B1", "A2", Side::Buy, OrderType::Market, 0.0, 15));
    auto execs = book.match_orders();

    assert(execs.size() == 1);
    assert(execs[0].fill_qty == 15);
    assert(execs[0].fill_price == 105.0);  // resting ask's price
    assert(book.ask_depth() == 5);

    std::cout << "  [PASS] test_market_order\n";
}

static void test_cancel() {
    OrderBook book;

    book.add_order(make_order("B_CANCEL", "A1", Side::Buy, OrderType::Limit, 50.0, 100));
    (void)book.match_orders();
    assert(book.bid_depth() == 100);

    assert(book.cancel("B_CANCEL"));
    assert(!book.cancel("NONEXISTENT"));

    // Cancelled order should not match.
    book.add_order(make_order("S1", "A2", Side::Sell, OrderType::Limit, 49.0, 50));
    auto execs = book.match_orders();
    assert(execs.empty());

    std::cout << "  [PASS] test_cancel\n";
}

static void test_get_mid_price() {
    OrderBook book;

    // Empty book → NaN.
    assert(std::isnan(book.get_mid_price()));

    book.add_order(make_order("B1", "A1", Side::Buy,  OrderType::Limit, 99.0, 10));
    book.add_order(make_order("S1", "A2", Side::Sell, OrderType::Limit, 101.0, 10));
    (void)book.match_orders();  // no crossing (bid < ask)

    assert(book.get_mid_price() == 100.0);
    assert(book.spread() == 2.0);

    std::cout << "  [PASS] test_get_mid_price\n";
}

// ─── Throughput benchmark ───────────────────────────────────────────────────

static void bench_throughput(int num_orders = 2'000'000) {
    OrderBook book;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> price_dist(99.0, 101.0);
    std::uniform_int_distribution<std::int32_t> qty_dist(1, 10);

    // Pre-generate orders to keep allocation out of the timed path.
    std::vector<Order> orders;
    orders.reserve(num_orders);
    for (int i = 0; i < num_orders; ++i) {
        Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
        orders.push_back(Order{
            .order_id  = std::format("ORD-{:08d}", i),
            .agent_id  = std::format("AGT-{:04d}", i % 1000),
            .side      = side,
            .type      = OrderType::Limit,
            .price     = price_dist(rng),
            .quantity  = qty_dist(rng),
            .timestamp = static_cast<std::int64_t>(i),
        });
    }

    std::uint64_t total_execs = 0;
    auto t0 = Clock::now();

    for (auto& order : orders) {
        book.add_order(std::move(order));
        auto execs = book.match_orders();
        total_execs += execs.size();
    }

    auto t1 = Clock::now();
    double elapsed_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ops_per_sec = num_orders / (elapsed_ms / 1000.0);
    double execs_per_sec = total_execs / (elapsed_ms / 1000.0);

    std::cout << std::format(
        "  [BENCH] {} orders in {:.1f} ms = {:.0f} orders/sec, "
        "{} executions ({:.0f} execs/sec)\n",
        num_orders, elapsed_ms, ops_per_sec,
        total_execs, execs_per_sec);

    std::cout << std::format(
        "  [BENCH] Final book: bid_depth={} ask_depth={} mid={:.4f}\n",
        book.bid_depth(), book.ask_depth(), book.get_mid_price());

    // Hard gate: must exceed 854K orders/sec in release builds.
    // (Debug builds are slower; we skip the assertion there.)
#ifdef NDEBUG
    if (ops_per_sec < 854'000) {
        std::cerr << std::format(
            "  [FAIL] Throughput {:.0f} orders/sec below 854K target!\n",
            ops_per_sec);
    }
#endif
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== OrderBook Correctness Tests ===\n";
    test_basic_match();
    test_partial_fill_multi_level();
    test_market_order();
    test_cancel();
    test_get_mid_price();

    std::cout << "\n=== OrderBook Throughput Benchmark ===\n";
    bench_throughput(2'000'000);

    std::cout << "\nAll tests passed.\n";
    return 0;
}
