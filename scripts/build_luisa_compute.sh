#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${GOB_LUISA_SOURCE_DIR:-${ROOT_DIR}/3rdparty/luisa_compute}"
BUILD_DIR="${GOB_LUISA_BUILD_DIR:-${ROOT_DIR}/build/luisa_compute}"
INSTALL_DIR="${GOB_LUISA_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CC_BIN="${GOB_LUISA_CC:-}"
CXX_BIN="${GOB_LUISA_CXX:-}"

# GCC 12.3 on Ubuntu 22.04 has a reproducible ICE while compiling Luisa's
# CUDAAccel generic lambda. GCC 11 is supported by Luisa and avoids the bug.
if [[ -z "${CC_BIN}" && -z "${CXX_BIN}" ]] && \
   command -v gcc-11 >/dev/null 2>&1 && command -v g++-11 >/dev/null 2>&1; then
    CC_BIN="$(command -v gcc-11)"
    CXX_BIN="$(command -v g++-11)"
fi
CC_BIN="${CC_BIN:-$(command -v cc)}"
CXX_BIN="${CXX_BIN:-$(command -v c++)}"

if [[ ! -f "${SOURCE_DIR}/CMakeLists.txt" ]]; then
    echo "LuisaCompute source is missing: ${SOURCE_DIR}" >&2
    echo "Run: git submodule update --init 3rdparty/luisa_compute" >&2
    exit 1
fi

for dependency in reproc spdlog; do
    if [[ ! -f "${SOURCE_DIR}/src/ext/${dependency}/CMakeLists.txt" ]]; then
        echo "LuisaCompute dependency is missing: src/ext/${dependency}" >&2
        echo "Run: git -C 3rdparty/luisa_compute submodule update --init --recursive" >&2
        exit 1
    fi
done

CMAKE_VERSION="$(${CMAKE_BIN} --version | sed -n '1s/^cmake version //p')"
if [[ "$(printf '%s\n' 3.26.0 "${CMAKE_VERSION}" | sort -V | head -n1)" != "3.26.0" ]]; then
    echo "LuisaCompute requires CMake 3.26 or newer; found ${CMAKE_VERSION}." >&2
    exit 1
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
fi

"${CMAKE_BIN}" -S "${SOURCE_DIR}" -B "${BUILD_DIR}" "${GENERATOR_ARGS[@]}" \
    -DCMAKE_C_COMPILER="${CC_BIN}" \
    -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DLUISA_COMPUTE_ENABLE_DSL=ON \
    -DLUISA_COMPUTE_ENABLE_CUDA=ON \
    -DLUISA_COMPUTE_ENABLE_DX=OFF \
    -DLUISA_COMPUTE_ENABLE_METAL=OFF \
    -DLUISA_COMPUTE_ENABLE_HIP=OFF \
    -DLUISA_COMPUTE_ENABLE_VULKAN=OFF \
    -DLUISA_COMPUTE_ENABLE_CPU=OFF \
    -DLUISA_COMPUTE_ENABLE_FALLBACK=OFF \
    -DLUISA_COMPUTE_ENABLE_REMOTE=OFF \
    -DLUISA_COMPUTE_ENABLE_GUI=OFF \
    -DLUISA_COMPUTE_ENABLE_RUST=OFF \
    -DLUISA_COMPUTE_ENABLE_TENSOR=OFF \
    -DLUISA_COMPUTE_ENABLE_CLANG_CXX=OFF \
    -DLUISA_COMPUTE_USE_SYSTEM_STL=ON \
    -DLUISA_COMPUTE_BUILD_TESTS=OFF \
    -DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF \
    -DLUISA_COMPUTE_DOWNLOAD_NVCOMP=OFF

"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
"${CMAKE_BIN}" --install "${BUILD_DIR}"

echo "LuisaCompute installed to ${INSTALL_DIR}"
echo "Configure Gobot with:"
echo "  -DGOB_BUILD_LUISA_RENDERER=ON -DGOB_LUISA_COMPUTE_ROOT=${INSTALL_DIR}"
