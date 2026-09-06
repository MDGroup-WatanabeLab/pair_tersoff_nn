#!/usr/bin/env bash
set -eu

# Edit these paths before running the script.
DIRECT_LAMMPS_BIN="/path/to/direct-lammps/lmp"
TORCHSCRIPT_LAMMPS_BIN="/path/to/torchscript-lammps/lmp"

cd "$(dirname "$0")"
"$DIRECT_LAMMPS_BIN" -in in.npt -var backend direct -log log.npt.direct
"$TORCHSCRIPT_LAMMPS_BIN" -in in.npt -var backend torchscript -log log.npt.torchscript
