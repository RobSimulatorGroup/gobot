#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${GOB_GSPLAT_SOURCE_DIR:-${ROOT_DIR}/3rdparty/gsplat_inference}"
BUILD_DIR="${GOB_GSPLAT_BUILD_DIR:-${ROOT_DIR}/build/gsplat_inference}"
INSTALL_DIR="${GOB_GSPLAT_INSTALL_DIR:-${BUILD_DIR}/install}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_JOBS="${BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}}"
CUDA_ARCHITECTURES="${GOBOT_GSPLAT_CUDA_ARCHITECTURES:-75-real;80-real;86-real;89-real;90-real}"

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
fi

CMAKE_VERSION="$(${CMAKE_BIN} --version | sed -n '1s/^cmake version //p')"
if [[ "$(printf '%s\n' 3.26.0 "${CMAKE_VERSION}" | sort -V | head -n1)" != "3.26.0" ]]; then
    echo "gsplat CUDA inference requires CMake 3.26 or newer; found ${CMAKE_VERSION}." >&2
    exit 1
fi

"${CMAKE_BIN}" -S "${SOURCE_DIR}" -B "${BUILD_DIR}" "${GENERATOR_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DGOBOT_GSPLAT_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
"${CMAKE_BIN}" --install "${BUILD_DIR}"

echo "gsplat CUDA inference library installed to ${INSTALL_DIR}"
