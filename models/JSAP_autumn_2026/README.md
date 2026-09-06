# sample_model_JSAP_autumn2026

Sample silicon model package for the Tersoff-NN LAMMPS interface presented at the JSAP Autumn 2026 meeting.

This sample model is included in the `pair_tersoff_nn` repository. It contains both runtime representations generated from the same checkpoint:

| File | Purpose |
| --- | --- |
| `model_ts.pt` | TorchScript representation for `tersoff_nn` |
| `model_direct.txt` | Direct C++ representation for `tersoff_nn_direct` |
| `input_stats.csv` | Input-feature normalization statistics |
| `Si.tersoff` | Matching Tersoff parameters, including the `Si_NN` entry |

## Use

Build the desired backend, then use this directory as the model package.

```bash
./examples/npt/run.sh
```

The corresponding LAMMPS definitions are:

```lammps
# TorchScript
pair_style tersoff_nn model_name model_ts.pt input_stats input_stats.csv use_gpu no
pair_coeff * * Si.tersoff Si_NN

# Direct C++
pair_style tersoff_nn_direct model_name model_direct.txt input_stats input_stats.csv use_gpu no
pair_coeff * * Si.tersoff Si_NN
```

Use absolute paths or paths relative to the LAMMPS working directory in an actual input file.

Example logs for both backends are stored in `../../examples/npt/`.

## License

`model_ts.pt`, `model_direct.txt`, and `input_stats.csv` are distributed under the MIT License. See `LICENSE-MIT`.

`Si.tersoff` is derived from the potential file distributed with LAMMPS and is distributed under the GNU General Public License version 2. Its original provenance and citation comments are retained in the file.
