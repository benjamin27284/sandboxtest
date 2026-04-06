// ============================================================================
// LOB Engine — Kafka-connected main loop.
//
// Architecture
// ────────────
//   1. KafkaConsumer reads serialised abms.market.Order from `orders_submit`.
//   2. Proto bytes → internal abms::lob::Order → OrderBook::add_order().
//   3. OrderBook::match_orders() → std::vector<Execution>.
//   4. Each fill is serialised to abms.market.Execution → `executions` topic.
//   5. Every 100 ms a MarketSnapshot is published to `market_data`.
//
// Build modes
// ───────────
//   Docker (production):  -DHAS_RDKAFKA=1 -DHAS_PROTOBUF=1
//                         Links librdkafka++ and libprotobuf.
//   Local  (dev/bench):   Neither defined.  Compiles with inert stubs so
//                         the benchmark and unit tests still run.
//
// Dependencies: librdkafka (C++ API), protobuf (>=3.x), C++20
// ============================================================================

#include "orderbook.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// ── Conditional includes for production dependencies ────────────────────────

#if defined(HAS_RDKAFKA)
#include <librdkafka/rdkafkacpp.h>
#endif

#if defined(HAS_PROTOBUF)
#include "market.pb.h"
#endif

// ── Globals ─────────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

static std::string env_or(const char* name, const char* fallback) {
    const char* val = std::getenv(name);
    return val ? std::string(val) : std::string(fallback);
}

// ============================================================================
// Production path: real librdkafka + protobuf
// ============================================================================

#if defined(HAS_RDKAFKA) && defined(HAS_PROTOBUF)

// ── Proto ↔ internal conversion ─────────────────────────────────────────────

static abms::lob::Order proto_to_order(const abms::market::Order& pb) {
    abms::lob::Order o;
    o.order_id  = pb.order_id();
    o.agent_id  = pb.agent_id();
    o.side      = (pb.side() == abms::market::BUY)
                      ? abms::lob::Side::Buy
                      : abms::lob::Side::Sell;
    o.type      = (pb.type() == abms::market::LIMIT)
                      ? abms::lob::OrderType::Limit
                      : abms::lob::OrderType::Market;
    o.price     = pb.price();
    o.quantity  = pb.quantity();
    o.timestamp = pb.timestamp();
    return o;
}

static std::string execution_to_proto(const abms::lob::Execution& exec,
                                      abms::lob::Side aggressor_side) {
    abms::market::Execution pb;
    pb.set_exec_id(exec.exec_id);
    pb.set_order_id(exec.order_id);
    pb.set_agent_id(exec.agent_id);
    pb.set_fill_price(exec.fill_price);
    pb.set_fill_qty(exec.fill_qty);
    pb.set_timestamp(exec.timestamp);
    pb.set_counter_order_id(exec.counter_order_id);
    pb.set_counter_agent_id(exec.counter_agent_id);
    pb.set_aggressor_side(aggressor_side == abms::lob::Side::Buy
                              ? abms::market::BUY
                              : abms::market::SELL);
    return pb.SerializeAsString();
}

static std::string snapshot_to_proto(const abms::lob::BookSnapshot& snap,
                                     std::int64_t tick = 0) {
    abms::market::MarketSnapshot pb;
    pb.set_mid_price(snap.mid_price);
    pb.set_spread(snap.spread);
    pb.set_total_volume(snap.total_volume);

    // Embed full OrderBookSnapshot for downstream consumers
    auto* book = pb.mutable_book_snapshot();
    book->set_tick(tick);
    book->set_mid_price(snap.mid_price);
    book->set_best_bid(snap.best_bid);
    book->set_best_ask(snap.best_ask);
    book->set_spread(snap.spread);
    for (const auto& lvl : snap.bid_levels) {
        auto* pl = book->add_bids();
        pl->set_price(lvl.price);
        pl->set_quantity(lvl.qty);
        pl->set_count(lvl.count);
    }
    for (const auto& lvl : snap.ask_levels) {
        auto* pl = book->add_asks();
        pl->set_price(lvl.price);
        pl->set_quantity(lvl.qty);
        pl->set_count(lvl.count);
    }
    return pb.SerializeAsString();
}

// ── Kafka helpers ───────────────────────────────────────────────────────────

/// Delivery-report callback — logs errors, silently acks successes.
class DeliveryReportCb : public RdKafka::DeliveryReportCb {
public:
    void dr_cb(RdKafka::Message& message) override {
        if (message.err()) {
            std::cerr << "[kafka] delivery failed: " << message.errstr()
                      << " (topic=" << message.topic_name() << ")\n";
        }
    }
};

/// Rebalance callback — log partition assignment for observability.
class RebalanceCb : public RdKafka::RebalanceCb {
public:
    void rebalance_cb(RdKafka::KafkaConsumer* consumer,
                      RdKafka::ErrorCode err,
                      std::vector<RdKafka::TopicPartition*>& partitions) override {
        if (err == RdKafka::ERR__ASSIGN_PARTITIONS) {
            std::cout << "[kafka] assigned " << partitions.size() << " partition(s)\n";
            consumer->assign(partitions);
        } else {
            std::cout << "[kafka] revoked " << partitions.size() << " partition(s)\n";
            consumer->unassign();
        }
    }
};

/// Build an RdKafka::Conf with common settings.
static std::unique_ptr<RdKafka::Conf> make_conf(const std::string& brokers) {
    std::string errstr;
    auto conf = std::unique_ptr<RdKafka::Conf>(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

    conf->set("bootstrap.servers",       brokers, errstr);
    conf->set("enable.auto.commit",      "true",  errstr);
    conf->set("auto.commit.interval.ms", "1000",  errstr);
    conf->set("session.timeout.ms",      "10000", errstr);
    return conf;
}

// ── Main (production) ───────────────────────────────────────────────────────

int main() {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // ── Configuration from environment ──────────────────────────────────────
    const auto brokers      = env_or("KAFKA_BROKERS",     "localhost:9092");
    const auto group_id     = env_or("KAFKA_GROUP_ID",    "lob-engine");
    const auto orders_topic = env_or("ORDERS_TOPIC",      "orders_submit");
    const auto exec_topic   = env_or("EXECUTIONS_TOPIC",  "executions");
    const auto market_topic = env_or("MARKET_DATA_TOPIC", "market_data");

    const int consumer_poll_ms = std::atoi(
        env_or("CONSUMER_POLL_MS", "1").c_str());  // 1 ms default for latency
    const int snapshot_interval_ms = std::atoi(
        env_or("SNAPSHOT_INTERVAL_MS", "100").c_str());

    // ── Consumer setup ──────────────────────────────────────────────────────
    std::string errstr;

    auto consumer_conf = make_conf(brokers);
    consumer_conf->set("group.id", group_id, errstr);

    static RebalanceCb rebalance_cb;
    consumer_conf->set("rebalance_cb", &rebalance_cb, errstr);

    std::unique_ptr<RdKafka::KafkaConsumer> consumer(
        RdKafka::KafkaConsumer::create(consumer_conf.get(), errstr));
    if (!consumer) {
        std::cerr << "[FATAL] consumer creation failed: " << errstr << "\n";
        return 1;
    }

    if (consumer->subscribe({orders_topic}) != RdKafka::ERR_NO_ERROR) {
        std::cerr << "[FATAL] subscribe to '" << orders_topic << "' failed\n";
        return 1;
    }

    // ── Producer setup ──────────────────────────────────────────────────────
    auto producer_conf = make_conf(brokers);

    static DeliveryReportCb dr_cb;
    producer_conf->set("dr_cb", &dr_cb, errstr);
    // Batch settings tuned for throughput
    producer_conf->set("linger.ms",            "5",     errstr);
    producer_conf->set("batch.num.messages",   "10000", errstr);
    producer_conf->set("queue.buffering.max.kbytes", "1048576", errstr);  // 1 GB

    std::unique_ptr<RdKafka::Producer> producer(
        RdKafka::Producer::create(producer_conf.get(), errstr));
    if (!producer) {
        std::cerr << "[FATAL] producer creation failed: " << errstr << "\n";
        return 1;
    }

    // ── Engine + bookkeeping ────────────────────────────────────────────────
    abms::lob::OrderBook book;
    std::int64_t snapshot_tick = 0;  // monotonic snapshot counter

    using SteadyClock = std::chrono::steady_clock;
    const auto snapshot_interval = std::chrono::milliseconds(snapshot_interval_ms);
    auto last_snapshot = SteadyClock::now();
    auto last_stats    = SteadyClock::now();

    std::cout << "=== LOB Engine started ===\n"
              << "  Kafka brokers:      " << brokers       << "\n"
              << "  Orders topic:       " << orders_topic  << "\n"
              << "  Executions topic:   " << exec_topic    << "\n"
              << "  Market data topic:  " << market_topic  << "\n"
              << "  Snapshot interval:  " << snapshot_interval_ms << " ms\n";

    // ── Main loop ───────────────────────────────────────────────────────────

    while (g_running.load(std::memory_order_relaxed)) {

        // 1. Consume incoming order from Kafka ───────────────────────────────
        std::unique_ptr<RdKafka::Message> msg(
            consumer->consume(consumer_poll_ms));

        if (msg->err() == RdKafka::ERR_NO_ERROR) {
            // 2. Deserialise protobuf → internal Order ───────────────────────
            abms::market::Order pb_order;
            if (pb_order.ParseFromArray(msg->payload(), static_cast<int>(msg->len()))) {
                abms::lob::Order order = proto_to_order(pb_order);

                // 3. Feed into the matching engine ───────────────────────────
                book.add_order(std::move(order));
                auto fills = book.match_orders();

                // 4. Serialise each fill → publish to executions topic ───────
                for (const auto& exec : fills) {
                    std::string payload = execution_to_proto(exec, exec.aggressor_side);
                    RdKafka::ErrorCode err = producer->produce(
                        exec_topic,
                        RdKafka::Topic::PARTITION_UA,
                        RdKafka::Producer::RK_MSG_COPY,
                        const_cast<char*>(payload.data()),
                        payload.size(),
                        /*key=*/exec.exec_id.data(),
                        exec.exec_id.size(),
                        /*timestamp=*/0,
                        /*msg_opaque=*/nullptr);

                    if (err != RdKafka::ERR_NO_ERROR) {
                        std::cerr << "[kafka] produce execution failed: "
                                  << RdKafka::err2str(err) << "\n";
                    }
                }
            } else {
                std::cerr << "[WARN] failed to parse Order proto ("
                          << msg->len() << " bytes)\n";
            }
        }
        // ERR__TIMED_OUT / ERR__PARTITION_EOF are normal — no action needed.

        // Serve the producer's delivery-report queue.
        producer->poll(0);

        // 5. Publish MarketSnapshot every 100 ms ────────────────────────────
        auto now = SteadyClock::now();
        if (now - last_snapshot >= snapshot_interval) {
            last_snapshot = now;

            auto snap = book.snapshot(/*depth=*/10);
            std::string payload = snapshot_to_proto(snap);

            RdKafka::ErrorCode err = producer->produce(
                market_topic,
                RdKafka::Topic::PARTITION_UA,
                RdKafka::Producer::RK_MSG_COPY,
                const_cast<char*>(payload.data()),
                payload.size(),
                /*key=*/nullptr, /*key_len=*/0,
                /*timestamp=*/0,
                /*msg_opaque=*/nullptr);

            if (err != RdKafka::ERR_NO_ERROR) {
                std::cerr << "[kafka] produce snapshot failed: "
                          << RdKafka::err2str(err) << "\n";
            }
        }

        // Periodic stats logging (every 10 s)
        if (now - last_stats >= std::chrono::seconds(10)) {
            last_stats = now;
            std::cout << "[LOB] orders=" << book.total_orders_processed()
                      << " execs=" << book.total_executions()
                      << " mid=" << book.get_mid_price()
                      << " spread=" << book.spread()
                      << " depth=" << book.total_volume()
                      << "\n";
        }
    }

    // ── Graceful shutdown ───────────────────────────────────────────────────
    std::cout << "\n[LOB] Shutting down...\n";

    consumer->close();

    // Flush outstanding produce requests (up to 5 s).
    producer->flush(5000);

    std::cout << "=== LOB Engine stopped ===\n"
              << "  Total orders processed: " << book.total_orders_processed() << "\n"
              << "  Total executions:       " << book.total_executions()       << "\n";

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}

// ============================================================================
// Stub path: local dev builds without rdkafka / protobuf
// ============================================================================

#else  // !defined(HAS_RDKAFKA) || !defined(HAS_PROTOBUF)

int main() {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "=== LOB Engine (stub mode — no Kafka / Protobuf) ===\n"
              << "  Build with -DHAS_RDKAFKA=1 -DHAS_PROTOBUF=1 for production.\n"
              << "  Run lob_bench for matching-engine validation.\n\n";

    abms::lob::OrderBook book;

    // Idle loop so the binary is runnable for smoke-testing.
    while (g_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\n=== LOB Engine (stub) stopped ===\n";
    return 0;
}

#endif
