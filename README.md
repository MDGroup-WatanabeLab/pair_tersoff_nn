# pair_tersoff_nn

LAMMPS plugins for the Tersoff-NN potential.

Two inference backends are provided.

| Backend     | Pair style          | Model file         | PyTorch dependency | GPU      |
| ----------- | ------------------- | ------------------ | ------------------ | -------- |
| Direct C++  | `tersoff_nn_direct` | `model_direct.txt` | No                 | No       |
| TorchScript | `tersoff_nn`        | `model_ts.pt`      | LibTorch           | Optional |

Both backends use the same Tersoff-NN model parameters and feature normalization.

## Repository structure

```text
pair_tersoff_nn/
├── direct/          # Direct C++ inference
├── torchscript/     # LibTorch-based inference
├── examples/
│   ├── npt/                 # 300 K, 0 bar NPT example
│   ├── energy_convergence/  # NVE time-step convergence
│   └── liquid_Si/           # 3000 K silicon melting example
└── models/
    └── JSAP_autumn_2026/  # Sample Si model
```

## Requirements

* Linux
* C++17 compiler
* CMake 3.16 or newer
* LAMMPS built with `PKG_PLUGIN=yes`
* PyTorch or a standalone LibTorch distribution for the TorchScript backend

The current implementation has been tested with LAMMPS 2 Apr 2025 and 10 Sep 2025.

## Build

Each backend can be built independently.

```bash
./direct/build.sh
```

or

```bash
conda activate tersoff_nn
./torchscript/build.sh
```

Set `LAMMPS_SOURCE_DIR` in each `build.sh` before compilation.
The TorchScript build defaults to `USE_GPU="no"` and discovers PyTorch from the active Python environment. To build without CUDA, use a CPU-only PyTorch installation. To build against a standalone LibTorch distribution, set `TORCH_CMAKE_PREFIX` in `torchscript/build.sh` to its absolute path. For a CUDA-enabled build that needs an explicit compiler path, set `USE_GPU="yes"` and `CUDA_COMPILER`.

The generated plugins are

```text
direct/build/tersoffnndirectplugin.so
torchscript/build/tersoffnnplugin.so
```

## Usage

### Direct C++

This pair style is derived from the original LAMMPS `pair_style tersoff` implementation and modified to evaluate the Tersoff-NN bond order with direct C++ inference.

```lammps
plugin load /path/to/tersoffnndirectplugin.so

pair_style tersoff_nn_direct model_name model_direct.txt input_stats input_stats.csv use_gpu no
pair_coeff * * Si.tersoff Si_NN
```

### TorchScript

This pair style is derived from the original LAMMPS `pair_style tersoff` implementation and modified to evaluate the Tersoff-NN bond order through LibTorch.

```lammps
plugin load /path/to/tersoffnnplugin.so

pair_style tersoff_nn model_name model_ts.pt input_stats input_stats.csv use_gpu no
pair_coeff * * Si.tersoff Si_NN
```

Set `use_gpu yes` for the TorchScript backend when using CUDA-enabled LibTorch.

## Model files

A model package contains

```text
si_model/
├── model_ts.pt
├── model_direct.txt
├── input_stats.csv
└── Si.tersoff
```

`model_ts.pt` and `model_direct.txt` must be generated from the same trained model.

## MD example

The included example runs 1000 steps of NPT molecular dynamics for 64-atom diamond Si at 300 K and 0 bar with both backends.

```bash
./examples/npt/run.sh
```

Set the two LAMMPS executable paths at the top of the script before running.
The example creates no dump files and records the density in each log file.

The NVE example compares energy conservation at time steps of 1.0, 0.5, and 0.1 fs using the Direct C++ backend.

```bash
./examples/energy_convergence/run.sh
```

The maximum absolute energy deviation from the initial value was

| Time step | Maximum deviation |
| --- | ---: |
| 1.0 fs | `3.83e-5 eV/atom` |
| 0.5 fs | `9.56e-6 eV/atom` |
| 0.1 fs | `3.92e-7 eV/atom` |

The liquid-Si example runs 1000 steps of NPT molecular dynamics for 512-atom silicon at 3000 K and 0 bar with both backends.

```bash
./examples/liquid_Si/run.sh
```

This short run is a melting demonstration and smoke test, not an equilibrated liquid-production trajectory.
The mean coordination number within 3.0 Angstrom increases from 4.0 to approximately 5.2 during the run.

## License

Both pair-style implementations are derived from the original LAMMPS `pair_style tersoff` source and are distributed under the GNU General Public License version 2. The original copyright and license notices are retained in both implementations. See `LICENSE`.

The learned model files and normalization statistics in `models` are distributed separately under the MIT License. The accompanying `Si.tersoff` file remains under the GNU General Public License version 2 as described in the model package README.

## Contact

Yusuke Nishimura: yusukeskelton@toki.waseda.jp
