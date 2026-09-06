#!/usr/bin/env bash
set -eu

# Edit this path before running the script.
LAMMPS_SOURCE_DIR="/path/to/direct-lammps/src"

cd "$(dirname "$0")"
rm -rf build
cmake -S . -B build \
  -D LAMMPS_SOURCE_DIR="$LAMMPS_SOURCE_DIR" \
  -D CMAKE_BUILD_TYPE=Release
cmake --build build
