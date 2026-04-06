cmake_minimum_required(VERSION 3.20)
project(lob-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Release flags: -O3 for hot-path throughput ─────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── Protobuf code generation ──────────────────────────────────────────────
# Generates market.pb.h / market.pb.cc from proto/market.proto.
# Only available when protoc + libprotobuf-dev are installed (Docker build).

find_package(Protobuf QUIET)
if(Protobuf_FOUND)
    message(STATUS "Protobuf found: ${Protobuf_VERSION}")
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/market.proto)
    add_library(market_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
    target_include_directories(market_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(market_proto PUBLIC protobuf::libprotobuf)
    set(HAS_PROTO TRUE)
else()
    message(STATUS "Protobuf not found — building without proto serialisation")
    set(HAS_PROTO FALSE)
endif()

# ── librdkafka ─────────────────────────────────────────────────────────────

find_package(RdKafka QUIET)
if(RdKafka_FOUND)
    message(STATUS "librdkafka found")
else()
    message(STATUS "librdkafka not found — building with Kafka stubs")
endif()

# ── Core matching-engine library (zero external deps) ─────────────────────

add_library(lob_core STATIC
    src/orderbook.cpp
)
target_include_directories(lob_core PUBLIC include)

# ── Main executable ────────────────────────────────────────────────────────

add_executable(lob_engine src/main.cpp)
target_link_libraries(lob_engine PRIVATE lob_core)

if(RdKafka_FOUND AND HAS_PROTO)
    target_link_libraries(lob_engine PRIVATE RdKafka::rdkafka++ market_proto)
    target_compile_definitions(lob_engine PRIVATE HAS_RDKAFKA=1 HAS_PROTOBUF=1)
elseif(RdKafka_FOUND)
    target_link_libraries(lob_engine PRIVATE RdKafka::rdkafka++)
    target_compile_definitions(lob_engine PRIVATE HAS_RDKAFKA=1)
elseif(HAS_PROTO)
    target_link_libraries(lob_engine PRIVATE market_proto)
    target_compile_definitions(lob_engine PRIVATE HAS_PROTOBUF=1)
endif()

# ── Tests / Benchmarks ────────────────────────────────────────────────────

enable_testing()

add_executable(lob_bench tests/bench_orderbook.cpp)
target_link_libraries(lob_bench PRIVATE lob_core)
add_test(NAME lob_bench COMMAND lob_bench)
