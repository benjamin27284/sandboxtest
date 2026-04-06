# =============================================================================
# LOB Engine — multi-stage C++ build
# =============================================================================
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build \
        librdkafka-dev libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
COPY proto/ proto/

RUN cmake -B out -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build out --parallel

# Run tests during build
RUN cd out && ctest --output-on-failure

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        librdkafka1 libprotobuf32t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/out/lob_engine /usr/local/bin/lob_engine

ENTRYPOINT ["lob_engine"]
