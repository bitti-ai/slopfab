# Keep the entire compiler/sysroot on the release glibc baseline.
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build python3-pip binutils libpng-dev libjpeg-dev \
    && python3 -m pip install --no-cache-dir cmake==3.31.6 \
    && rm -rf /var/lib/apt/lists/*
# Host C API tests load the library without a GPU. The toolkit's driver stub
# supplies its SONAME only inside this build container; it is never packaged.
RUN mkdir -p /opt/cuda-stubs \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
ENV LD_LIBRARY_PATH="/opt/cuda-stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
WORKDIR /src
CMD ["bash", "tools/build_linux_library.sh", "--build-dir", "/build/slopfab"]
