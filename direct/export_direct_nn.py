#!/usr/bin/env python3
"""Export a TorchScript bond-order model to the direct NN text format."""

from pathlib import Path

import torch
import torch.nn.functional as F


# Edit these paths when converting another model. The default assumes a
# separate tersoff_nn_models repository next to pair_tersoff_nn. The script
# intentionally uses ordinary Python configuration instead of command-line
# arguments.
DIRECT_DIR = Path(__file__).resolve().parent
MODEL_REPOSITORY = DIRECT_DIR.parent.parent / "tersoff_nn_models"
MODEL_NAME = "si_weakreg_finetune_total200"
MODEL_DIR = MODEL_REPOSITORY / MODEL_NAME
MODEL_PATH = MODEL_DIR / "model_ts.pt"
OUTPUT_PATH = MODEL_DIR / "model_direct.txt"

SUPPORTED_CUTOFF_FUNCTIONS = {"tersoff", "smooth"}
XI_LAYER_SPECS = [(0, "silu"), (2, "silu"), (4, "softplus")]
BIJ_LAYER_SPECS = [
    (0, "shifted_softplus"),
    (2, "shifted_softplus"),
    (4, "shifted_softplus"),
]


def tensor_values(tensor):
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def write_values(output_file, values):
    for start in range(0, len(values), 8):
        line = " ".join(f"{value:.17g}" for value in values[start : start + 8])
        output_file.write(line)
        output_file.write("\n")


def get_float_attr(module, name):
    value = getattr(module, name)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def collect_layers(params, prefix, activations):
    layers = []
    for index, activation in activations:
        weight_name = f"{prefix}.{index}.weight"
        bias_name = f"{prefix}.{index}.bias"
        if weight_name not in params:
            raise KeyError(f"Missing parameter: {weight_name}")
        weight = params[weight_name].detach().cpu()
        bias = params.get(bias_name)
        layers.append(
            (weight, None if bias is None else bias.detach().cpu(), activation)
        )
    return layers


def cutoff_value(distance, bigr, bigd, cutoff_function):
    if distance < bigr - bigd:
        return 1.0
    if distance > bigr + bigd:
        return 1.0e-7

    phase = torch.pi * (distance - bigr) / (2.0 * bigd)
    if cutoff_function == "tersoff":
        return 0.5 * (1.0 - torch.sin(torch.tensor(phase))).item()
    if cutoff_function == "smooth":
        return (
            0.5
            - 0.5625 * torch.sin(torch.tensor(phase)).item()
            - 0.0625 * torch.sin(torch.tensor(3.0 * phase)).item()
        )
    raise ValueError(f"Unsupported cutoff function: {cutoff_function}")


def infer_nn_cutoff_function(model, bigr_table, bigd_table):
    """Identify the cutoff implemented by fc_tensor for legacy models."""
    bigr = float(bigr_table.reshape(-1)[0])
    bigd = float(bigd_table.reshape(-1)[0])
    distance = bigr - 0.37 * bigd
    atom_types = torch.zeros((1, 3), dtype=torch.long)
    model_value = float(
        model.fc_tensor(torch.tensor([distance]), atom_types).detach().cpu()[0]
    )

    errors = {
        name: abs(model_value - cutoff_value(distance, bigr, bigd, name))
        for name in SUPPORTED_CUTOFF_FUNCTIONS
    }
    cutoff_function = min(errors, key=errors.get)
    if errors[cutoff_function] > 1.0e-5:
        raise ValueError(
            "Could not identify the cutoff implemented by model.fc_tensor; "
            f"best error was {errors[cutoff_function]:.6g}"
        )
    return cutoff_function


def cutoff_functions(model, bigr_table, bigd_table):
    if hasattr(model, "get_cutoff_function"):
        cutoff_function = str(model.get_cutoff_function())
        if cutoff_function not in SUPPORTED_CUTOFF_FUNCTIONS:
            raise ValueError(
                f"Unsupported cutoff_function stored in model: {cutoff_function}"
            )
        return cutoff_function, cutoff_function

    # This matches the fallback in tersoff_plugin/260714: the outer pair
    # potential uses Tersoff cutoff, while legacy TorchScript may contain a
    # different fc_tensor implementation for the NN zeta accumulation.
    return "tersoff", infer_nn_cutoff_function(model, bigr_table, bigd_table)


def validate_model_tables(embedding, bigr_table, bigd_table, p_table):
    num_types, embedding_dim = embedding.shape
    expected_triplets = num_types**3
    expected_pairs = num_types**2
    if bigr_table.numel() != expected_triplets:
        raise ValueError(
            f"bigr_tensor has {bigr_table.numel()} values; expected {expected_triplets}"
        )
    if bigd_table.numel() != expected_triplets:
        raise ValueError(
            f"bigd_tensor has {bigd_table.numel()} values; expected {expected_triplets}"
        )
    if p_table.numel() != expected_pairs:
        raise ValueError(
            f"p_table has {p_table.numel()} values; expected {expected_pairs}"
        )
    if embedding_dim <= 0:
        raise ValueError("atom_type_embedding must have a positive embedding dimension")


def validate_network_layout(params, embedding_dim, xi_layers, bij_layers):
    expected_parameters = {"atom_type_embedding.weight"}
    if "p_raw_table" in params:
        expected_parameters.add("p_raw_table")

    for prefix, specs in [
        ("network_xi.network", XI_LAYER_SPECS),
        ("network_bij.network", BIJ_LAYER_SPECS),
    ]:
        for index, _ in specs:
            weight_name = f"{prefix}.{index}.weight"
            bias_name = f"{prefix}.{index}.bias"
            expected_parameters.add(weight_name)
            if bias_name in params:
                expected_parameters.add(bias_name)

    unexpected = sorted(set(params) - expected_parameters)
    if unexpected:
        raise ValueError(
            "Unsupported trainable parameters in model: " + ", ".join(unexpected)
        )

    def validate_chain(name, layers, expected_input, expected_output):
        current_width = expected_input
        for layer_index, (weight, bias, _) in enumerate(layers):
            if weight.ndim != 2 or weight.shape[1] != current_width:
                raise ValueError(
                    f"{name} layer {layer_index} has shape {tuple(weight.shape)}; "
                    f"expected input width {current_width}"
                )
            if bias is not None and (
                bias.ndim != 1 or bias.shape[0] != weight.shape[0]
            ):
                raise ValueError(
                    f"{name} layer {layer_index} bias shape does not match its output"
                )
            current_width = weight.shape[0]
        if current_width != expected_output:
            raise ValueError(
                f"{name} output width is {current_width}; expected {expected_output}"
            )

    validate_chain("xi", xi_layers, 3 + 3 * embedding_dim, 1)
    validate_chain("bij", bij_layers, 1, 1)


def export_direct_nn(model_path=MODEL_PATH, output_path=OUTPUT_PATH):
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {model_path}")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    embedding = params["atom_type_embedding.weight"].detach().cpu()
    p_table = buffers.get("p_table")
    if p_table is None:
        p_raw = params.get("p_raw_table")
        if p_raw is None:
            raise KeyError("Missing p_table buffer or p_raw_table parameter")
        p_table = F.softplus(p_raw.detach()).cpu() + 1.0e-12

    bigr_table = buffers["bigr_tensor"].detach().cpu()
    bigd_table = buffers["bigd_tensor"].detach().cpu()
    validate_model_tables(embedding, bigr_table, bigd_table, p_table)

    pair_cutoff_function, nn_cutoff_function = cutoff_functions(
        model, bigr_table, bigd_table
    )

    xi_layers = collect_layers(
        params,
        "network_xi.network",
        XI_LAYER_SPECS,
    )
    bij_layers = collect_layers(
        params,
        "network_bij.network",
        BIJ_LAYER_SPECS,
    )
    validate_network_layout(params, embedding.shape[1], xi_layers, bij_layers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("tersoff_nn_direct_v2\n")
        output_file.write(f"pair_cutoff_function {pair_cutoff_function}\n")
        output_file.write(f"nn_cutoff_function {nn_cutoff_function}\n")
        output_file.write(f"num_types {embedding.shape[0]}\n")
        output_file.write(f"dim_embed_per_atom {embedding.shape[1]}\n")
        output_file.write(f"rmin {get_float_attr(model, 'rmin'):.17g}\n")
        output_file.write(f"rmax {get_float_attr(model, 'rmax'):.17g}\n")

        for name, tensor in [
            ("bigr_table", bigr_table),
            ("bigd_table", bigd_table),
            ("p_table", p_table),
            ("embedding", embedding),
        ]:
            values = tensor_values(tensor)
            output_file.write(f"{name} {len(values)}\n")
            write_values(output_file, values)

        def write_layers(name, layers):
            output_file.write(f"{name} {len(layers)}\n")
            for weight, bias, activation in layers:
                out_features, in_features = weight.shape
                has_bias = 0 if bias is None else 1
                output_file.write(
                    f"layer {out_features} {in_features} {has_bias} {activation}\n"
                )
                write_values(output_file, tensor_values(weight))
                if bias is not None:
                    write_values(output_file, tensor_values(bias))

        write_layers("xi_layers", xi_layers)
        write_layers("bij_layers", bij_layers)
        output_file.write("end\n")

    print(f"Source model: {model_path}")
    print(f"Pair cutoff function: {pair_cutoff_function}")
    print(f"NN cutoff function: {nn_cutoff_function}")
    print(f"Direct NN model written to: {output_path}")
    return output_path


def main():
    export_direct_nn()


if __name__ == "__main__":
    main()
