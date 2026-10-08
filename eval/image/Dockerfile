# Isolated build-and-run image for evaluating gptoss-infer commits. Maintainer-owned. Every evaluation runs it
# with --network none, read-only weights and data, and no credentials; it holds only the toolchain.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
