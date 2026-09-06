#!/usr/bin/env bash
set -eu

# yes: use CUDA_COMPILER explicitly, no: use the installed PyTorch configuration
USE_GPU="no"

# Edit these paths before running the script.
LAMMPS_SOURCE_DIR="/path/to/torchscript-lammps/src"
CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
# Leave empty to discover PyTorch from the active Python environment.
# For a standalone LibTorch distribution, set this to its absolute path.
TORCH_CMAKE_PREFIX=""

cd "$(dirname "$0")"
if [[ -z "$TORCH_CMAKE_PREFIX" ]]; then
  TORCH_CMAKE_PREFIX=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
fi
CUDA_OPTION=()
if [[ "$USE_GPU" == "yes" ]]; then
  CUDA_OPTION=(-D CMAKE_CUDA_COMPILER="$CUDA_COMPILER")
fi

rm -rf build
cmake -S . -B build \
  -D LAMMPS_SOURCE_DIR="$LAMMPS_SOURCE_DIR" \
  -D CMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
  "${CUDA_OPTION[@]}" \
  -D CMAKE_BUILD_TYPE=Release
cmake --build build
