# Keep the entire compiler/sysroot on the release glibc baseline.
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build python3-pip binutils libpng-dev libjpeg-dev \
    && python3 -m pip install --no-cache-dir cmake==3.31.6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
CMD ["bash", "tools/build_linux_library.sh", "--build-dir", "/build/slopfab"]
