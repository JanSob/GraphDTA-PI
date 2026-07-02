import argparse
import inspect
import json
import os

import pandas as pd
import torch
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

from dataset import TestbedDataset
from models.gat import GATNet
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from subgraph_explainer import (
    attach_target_fields,
    bitmask_to_atoms,
    deduplicate_sufficient_subgraphs_by_atoms,
    describe_subgraph_atoms,
    enumerate_connected_subgraphs,
    filter_minimal_sufficient_subgraphs,
    format_subgraph_components,
    generate_pairwise_disjoint_combinations,
    smile_to_mining_graph,
    subgraph_to_smiles,
)


MODELS = {
    "GINConvNet": GINConvNet,
    "GATNet": GATNet,
    "GCNNet": GCNNet,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def model_accepts_argument(model_class, argument_name):
    signature = inspect.signature(model_class.__init__)
    return argument_name in signature.parameters


def build_masked_candidate_data(mining_graph, subgraph_mask):
    atom_count = mining_graph["atom_count"]
    node_mask = torch.zeros((atom_count, 1), dtype=torch.float32)

    for atom_idx in bitmask_to_atoms(subgraph_mask):
        node_mask[atom_idx, 0] = 1.0

    graph_data = DATA.Data(
        x=mining_graph["full_x_tensor"],
        edge_index=mining_graph["edge_index_tensor"],
    )

    graph_data.node_mask = node_mask
    graph_data.mol_features = mining_graph["mol_features_tensor"]
    graph_data.__setitem__("c_size", torch.LongTensor([atom_count]))
    return graph_data


def prepare_data_for_model(data, protein_feat_dim):
    prepared = data.clone()

    if hasattr(prepared, "protein_feat") and prepared.protein_feat is not None:
        current_dim = prepared.protein_feat.view(prepared.protein_feat.size(0), -1).shape[1]

        if current_dim > protein_feat_dim:
            prepared.protein_feat = prepared.protein_feat[..., :protein_feat_dim]
        elif current_dim < protein_feat_dim:
            pad_width = protein_feat_dim - current_dim
            padding = torch.zeros(
                *prepared.protein_feat.shape[:-1],
                pad_width,
                dtype=prepared.protein_feat.dtype,
            )
            prepared.protein_feat = torch.cat((prepared.protein_feat, padding), dim=-1)

    return prepared


def predict_single(model, device, data):
    model.eval()
    loader = DataLoader([data], batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)

    return prediction.cpu().item()


def predict_many(model, device, data_list, batch_size=256):
    model.eval()
    predictions = []
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().view(-1).tolist())

    return predictions


def predict_candidate_batches(
    model,
    device,
    full_data,
    mining_graph,
    candidate_iter,
    protein_feat_dim,
    candidate_buffer_size=256,
    prediction_batch_size=256,
    progress_prefix="",
):
    model.eval()
    candidate_data_buffer = []
    candidate_record_buffer = []
    predicted_candidate_batches = 0

    def flush():
        nonlocal candidate_data_buffer
        nonlocal candidate_record_buffer
        nonlocal predicted_candidate_batches

        if not candidate_data_buffer:
            return

        predicted_candidate_batches += 1
        print(
            f"{progress_prefix}predicting candidate buffer "
            f"{predicted_candidate_batches} with {len(candidate_data_buffer)} candidates",
            flush=True,
        )

        predictions = predict_many(
            model,
            device,
            candidate_data_buffer,
            batch_size=prediction_batch_size,
        )
        current_records = candidate_record_buffer
        candidate_data_buffer = []
        candidate_record_buffer = []

        for record, prediction in zip(current_records, predictions):
            yield record, prediction

    for union_mask, components in candidate_iter:
        subgraph_data = build_masked_candidate_data(mining_graph, union_mask)
        subgraph_data = attach_target_fields(subgraph_data, full_data)
        subgraph_data = prepare_data_for_model(subgraph_data, protein_feat_dim)
        candidate_data_buffer.append(subgraph_data)
        candidate_record_buffer.append({
            "mask": union_mask,
            "components": components,
            "num_components": len(components),
            "size": union_mask.bit_count(),
        })

        if len(candidate_data_buffer) >= candidate_buffer_size:
            yield from flush()

    yield from flush()


def compute_indispensability(
    model,
    device,
    full_data,
    mining_graph,
    full_prediction,
    minimal_sufficient_subgraphs,
    protein_feat_dim,
    prediction_batch_size,
):
    full_mask = (1 << mining_graph["atom_count"]) - 1
    occluded_data_list = []

    for item in minimal_sufficient_subgraphs:
        occluded_mask = full_mask & ~item["mask"]
        occluded_data = build_masked_candidate_data(mining_graph, occluded_mask)
        occluded_data = attach_target_fields(occluded_data, full_data)
        occluded_data = prepare_data_for_model(occluded_data, protein_feat_dim)
        occluded_data_list.append(occluded_data)

    occluded_predictions = predict_many(
        model,
        device,
        occluded_data_list,
        batch_size=prediction_batch_size,
    )

    for item, occluded_prediction in zip(minimal_sufficient_subgraphs, occluded_predictions):
        indispensability_delta = full_prediction - occluded_prediction
        item["occluded_prediction"] = occluded_prediction
        item["indispensability_delta"] = indispensability_delta
        item["indispensability"] = abs(indispensability_delta)


def pick_best_candidate(candidates):
    if not candidates:
        return None

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            item["size"],
            item["difference"],
            -item["indispensability"],
            item["num_components"],
        ),
    )
    return ordered_candidates[0]


def explain_sample(
    model,
    device,
    full_data,
    protein_feat_dim,
    smiles,
    epsilon,
    max_subgraph_size,
    max_combination_size,
    max_total_atoms,
    candidate_buffer_size,
    prediction_batch_size,
):
    mining_graph = smile_to_mining_graph(smiles)
    full_data = prepare_data_for_model(full_data, protein_feat_dim)
    full_prediction = predict_single(model, device, full_data)

    subgraph_count = sum(
        1
        for _ in enumerate_connected_subgraphs(
            mining_graph["neighbor_map"],
            max_size=max_subgraph_size,
        )
    )
    print(
        f"    atom_count={mining_graph['atom_count']} connected_subgraphs={subgraph_count}",
        flush=True,
    )

    subgraphs = enumerate_connected_subgraphs(
        mining_graph["neighbor_map"],
        max_size=max_subgraph_size,
    )
    candidate_combinations = generate_pairwise_disjoint_combinations(
        subgraphs,
        max_combination_size=max_combination_size,
        max_total_atoms=max_total_atoms,
    )

    sufficient_subgraphs = []
    candidate_count = 0

    for record, subgraph_prediction in predict_candidate_batches(
        model,
        device,
        full_data,
        mining_graph,
        candidate_combinations,
        protein_feat_dim,
        candidate_buffer_size=candidate_buffer_size,
        prediction_batch_size=prediction_batch_size,
        progress_prefix="    ",
    ):
        candidate_count += 1
        if candidate_count == 1 or candidate_count % 5000 == 0:
            print(f"    processed candidates: {candidate_count}", flush=True)

        difference = abs(full_prediction - subgraph_prediction)
        if difference <= epsilon:
            sufficient_subgraphs.append({
                **record,
                "prediction": subgraph_prediction,
                "difference": difference,
            })

    print(
        f"    candidate combinations={candidate_count} sufficient={len(sufficient_subgraphs)}",
        flush=True,
    )

    sufficient_subgraphs = deduplicate_sufficient_subgraphs_by_atoms(
        sufficient_subgraphs
    )
    sufficient_subgraphs = sorted(
        sufficient_subgraphs,
        key=lambda item: (item["size"], item["difference"])
    )
    minimal_sufficient_subgraphs = filter_minimal_sufficient_subgraphs(
        sufficient_subgraphs
    )

    compute_indispensability(
        model,
        device,
        full_data,
        mining_graph,
        full_prediction,
        minimal_sufficient_subgraphs,
        protein_feat_dim,
        prediction_batch_size,
    )

    print(
        f"    minimal sufficient={len(minimal_sufficient_subgraphs)}",
        flush=True,
    )

    best_candidate = pick_best_candidate(minimal_sufficient_subgraphs)

    return {
        "full_prediction": full_prediction,
        "sufficient_count": len(sufficient_subgraphs),
        "minimal_count": len(minimal_sufficient_subgraphs),
        "best_candidate": best_candidate,
        "mining_graph": mining_graph,
    }


def serialize_best_candidate(best_candidate, mining_graph, true_affinity):
    if best_candidate is None:
        return {
            "pi_found": False,
            "pi_true_affinity": true_affinity,
            "pi_fragment_smiles": None,
            "pi_atom_indices": None,
            "pi_components": None,
            "pi_component_smiles": None,
            "pi_num_components": None,
            "pi_size": None,
            "pi_prediction": None,
            "pi_full_prediction_difference": None,
            "pi_occluded_prediction": None,
            "pi_indispensability_delta": None,
            "pi_indispensability": None,
            "pi_candidate_count": None,
            "full_prediction": None,
            "pi_atom_details": None,
        }

    components = best_candidate.get("components", (best_candidate["mask"],))

    return {
        "pi_found": True,
        "pi_true_affinity": true_affinity,
        "pi_fragment_smiles": subgraph_to_smiles(mining_graph, best_candidate["mask"]),
        "pi_atom_indices": json.dumps(bitmask_to_atoms(best_candidate["mask"])),
        "pi_components": json.dumps(format_subgraph_components(components)),
        "pi_component_smiles": json.dumps([
            subgraph_to_smiles(mining_graph, component)
            for component in components
        ]),
        "pi_num_components": best_candidate["num_components"],
        "pi_size": best_candidate["size"],
        "pi_prediction": best_candidate["prediction"],
        "pi_full_prediction_difference": best_candidate["difference"],
        "pi_occluded_prediction": best_candidate["occluded_prediction"],
        "pi_indispensability_delta": best_candidate["indispensability_delta"],
        "pi_indispensability": best_candidate["indispensability"],
        "pi_candidate_count": None,
        "full_prediction": None,
        "pi_atom_details": None,
    }


def infer_protein_feat_dim_from_state_dict(state_dict, fallback_dim):
    weight = state_dict.get("fc_protein_feat.weight")
    if weight is None:
        return fallback_dim
    return weight.shape[1]


def load_model(model_name, dataset, device, protein_feat_dim):
    model_class = MODELS[model_name]
    model_file = f"model_{model_name}_{dataset}.model"

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Could not find model file: {model_file}")

    state_dict = torch.load(model_file, map_location=device)
    protein_feat_dim = infer_protein_feat_dim_from_state_dict(
        state_dict,
        protein_feat_dim,
    )

    if model_accepts_argument(model_class, "protein_feat_dim"):
        model = model_class(protein_feat_dim=protein_feat_dim).to(device)
    else:
        model = model_class().to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, protein_feat_dim


def add_prefixed_columns(df, model_name, records):
    prefix = model_name.lower()
    for column_name in records[0].keys():
        df[f"{prefix}_{column_name}"] = [record[column_name] for record in records]


def build_pi_ready_df(df, model_name):
    prefix = model_name.lower()
    pi_found_col = f"{prefix}_pi_found"
    pi_smiles_col = f"{prefix}_pi_fragment_smiles"

    pi_ready_df = df.copy()
    pi_ready_df["original_compound_iso_smiles"] = pi_ready_df["compound_iso_smiles"]

    pi_ready_df = pi_ready_df[pi_ready_df[pi_found_col] == True].copy()
    pi_ready_df["compound_iso_smiles"] = pi_ready_df[pi_smiles_col]

    return pi_ready_df


def write_model_outputs(target_df, model_name, output_prefix):
    annotated_path = f"{output_prefix}_{model_name}.csv"
    pi_ready_path = f"{output_prefix}_{model_name}_pi_only.csv"

    target_df.to_csv(annotated_path, index=False)
    build_pi_ready_df(target_df, model_name).to_csv(pi_ready_path, index=False)

    return annotated_path, pi_ready_path


def pad_records_to_length(records, total_length):
    padded_records = list(records)
    while len(padded_records) < total_length:
        placeholder = serialize_best_candidate(None, None, None)
        placeholder["pi_found"] = None
        padded_records.append(placeholder)
    return padded_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate a test CSV with best-PI fidelity columns for each model."
    )
    parser.add_argument(
        "--dataset",
        default="davis",
        choices=["davis", "kiba"],
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to data/<dataset>_test_pi_fidelity.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GINConvNet", "GATNet", "GCNNet"],
        choices=sorted(MODELS.keys()),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive start index in the test set.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end index in the test set.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-subgraph-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-combination-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-total-atoms",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--candidate-buffer-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="If set, write one output CSV per model using this prefix.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Write per-model checkpoint CSVs every N processed samples.",
    )
    parser.add_argument(
        "--sample-indices-file",
        default=None,
        help="CSV file containing a sample_index column for explicit pair selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    csv_path = f"data/{args.dataset}_test.csv"
    output_path = args.output or f"data/{args.dataset}_test_pi_fidelity.csv"

    df = pd.read_csv(csv_path).copy()
    test_data = TestbedDataset(root="data", dataset=args.dataset + "_test")

    if args.sample_indices_file:
        sample_indices_df = pd.read_csv(args.sample_indices_file)
        if "sample_index" not in sample_indices_df.columns:
            raise ValueError("sample_indices_file must contain a sample_index column.")

        selected_indices = []
        seen_indices = set()
        for sample_index in sample_indices_df["sample_index"].tolist():
            sample_index = int(sample_index)
            if sample_index < 0 or sample_index >= len(df):
                raise ValueError(f"sample_index out of range: {sample_index}")
            if sample_index in seen_indices:
                continue
            seen_indices.add(sample_index)
            selected_indices.append(sample_index)
    else:
        start_index = args.start_index
        end_index = len(df) if args.end_index is None else min(args.end_index, len(df))

        if start_index < 0 or start_index >= end_index:
            raise ValueError("Invalid start/end index range.")

        selected_indices = list(range(start_index, end_index))

    selected_index_set = set(selected_indices)
    selected_df = df.iloc[selected_indices].copy().reset_index(drop=True)

    print("device:", device)
    print("dataset:", args.dataset)
    print("rows:", len(df))
    print("selected pairs:", len(selected_indices))
    if args.sample_indices_file:
        print("sample index file:", args.sample_indices_file)
    else:
        print("processing range:", start_index, end_index)

    if args.output_prefix:
        output_targets = {
            model_name: selected_df.copy() for model_name in args.models
        }
    else:
        output_targets = {
            model_name: selected_df for model_name in args.models
        }

    for model_name in args.models:
        print(f"\nRunning {model_name}...")
        protein_feat_dim = test_data[0].protein_feat.view(-1).shape[0]
        model, protein_feat_dim = load_model(
            model_name,
            args.dataset,
            device,
            protein_feat_dim,
        )

        model_records = []
        processed_samples = 0
        for selected_position, sample_index in enumerate(selected_indices):
            if selected_position == 0 or selected_position % 10 == 0:
                print(
                    f"  {model_name}: pair {selected_position + 1}/{len(selected_indices)} "
                    f"(sample_index={sample_index})",
                    flush=True,
                )

            full_data = test_data[sample_index]
            smiles = df.iloc[sample_index]["compound_iso_smiles"]
            true_affinity = float(df.iloc[sample_index]["affinity"])

            explanation_result = explain_sample(
                model=model,
                device=device,
                full_data=full_data,
                protein_feat_dim=protein_feat_dim,
                smiles=smiles,
                epsilon=args.epsilon,
                max_subgraph_size=args.max_subgraph_size,
                max_combination_size=args.max_combination_size,
                max_total_atoms=args.max_total_atoms,
                candidate_buffer_size=args.candidate_buffer_size,
                prediction_batch_size=args.prediction_batch_size,
            )

            best_candidate = explanation_result["best_candidate"]
            serialized = serialize_best_candidate(
                best_candidate,
                explanation_result["mining_graph"],
                true_affinity,
            )
            serialized["pi_candidate_count"] = explanation_result["minimal_count"]
            serialized["full_prediction"] = explanation_result["full_prediction"]
            serialized["pi_atom_details"] = (
                json.dumps(
                    describe_subgraph_atoms(
                        explanation_result["mining_graph"],
                        best_candidate["mask"],
                    )
                )
                if best_candidate is not None else None
            )

            model_records.append(serialized)
            processed_samples += 1

            if args.output_prefix and processed_samples % args.checkpoint_interval == 0:
                checkpoint_df = selected_df.copy()
                add_prefixed_columns(
                    checkpoint_df,
                    model_name,
                    pad_records_to_length(model_records, len(selected_df)),
                )
                annotated_path, pi_ready_path = write_model_outputs(
                    checkpoint_df,
                    model_name,
                    args.output_prefix,
                )
                print(
                    f"Checkpointed {model_name} after {processed_samples} processed samples "
                    f"to {annotated_path} and {pi_ready_path}",
                    flush=True,
                )

        target_df = output_targets[model_name]
        add_prefixed_columns(target_df, model_name, model_records)

        if args.output_prefix:
            annotated_path, pi_ready_path = write_model_outputs(
                target_df,
                model_name,
                args.output_prefix,
            )
            print(f"Saved {model_name} results to {annotated_path}")
            print(f"Saved {model_name} PI-only data to {pi_ready_path}")

    if not args.output_prefix:
        df.to_csv(output_path, index=False)
        print(f"\nSaved annotated test set to {output_path}")


if __name__ == "__main__":
    main()
