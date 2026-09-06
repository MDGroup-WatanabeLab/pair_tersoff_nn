#!/usr/bin/env bash
set -eu

# Edit this path before running the script.
LAMMPS_BIN="/path/to/direct-lammps/lmp"

cd "$(dirname "$0")"
"$LAMMPS_BIN" -in in.nve -var dt 0.0010 -log log.nve.1fs
"$LAMMPS_BIN" -in in.nve -var dt 0.0005 -log log.nve.0p5fs
"$LAMMPS_BIN" -in in.nve -var dt 0.0001 -log log.nve.0p1fs
