import csv
import os
import torch
import pandas as pd
import json

from torch_geometric.loader import DataLoader
from models.ginconv import GINConvNet
from dataset import TestbedDataset
from subgraph_explainer import (
    smile_to_mining_graph,
    enumerate_connected_subgraphs,
    bitmask_to_atoms,
    subgraph_to_pyg_data,
    attach_target_fields,
    filter_minimal_sufficient_subgraphs,
    describe_subgraph_atoms,
    subgraph_to_smiles,
    generate_pairwise_disjoint_combinations,
    deduplicate_sufficient_subgraphs_by_atoms,
    format_subgraph_components,
)

"""
Prototype script for ligand subgraph explanation mining.

Loads a trained GraphDTA model, generates ligand subgraph combinations,
predicts their affinity, and exports minimal sufficient fragments to CSV.
"""

def get_device():
    """Return the best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def predict_single(model, device, data):
    """Run model prediction for one PyG Data sample."""
    model.eval()
    loader = DataLoader([data], batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)

    return prediction.cpu().item()

def predict_many(model, device, data_list, batch_size=256):
    """Run batched model predictions for multiple PyG Data samples."""
    model.eval()
    predictions = []

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"Predicting batch {batch_index}/{total_batches}",
                    flush=True
                )

            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().view(-1).tolist())

    return predictions


def predict_candidate_batches(model, device, full_data, mining_graph, candidate_iter, batch_size=256):
    """Run batched model predictions for streaming candidate subgraphs."""
    model.eval()
    batch_data = []
    batch_records = []

    # batch candidate scoring so we don't store every Data object at once
    def flush():
        nonlocal batch_data, batch_records

        if not batch_data:
            return

        predictions = []
        loader = DataLoader(batch_data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch)
                predictions.extend(output.cpu().view(-1).tolist())

        current_records = batch_records
        batch_data = []
        batch_records = []

        for record, prediction in zip(current_records, predictions):
            yield record, prediction

    for union_mask, components in candidate_iter:
        subgraph_data = subgraph_to_pyg_data(mining_graph, union_mask)
        subgraph_data = attach_target_fields(subgraph_data, full_data)

        batch_data.append(subgraph_data)
        batch_records.append({
            "mask": union_mask,
            "components": components,
            "num_components": len(components),
            "size": union_mask.bit_count(),
        })

        if len(batch_data) >= batch_size:
            yield from flush()

    yield from flush()

def main():
    """Run the subgraph explanation prototype for one selected test sample."""
    dataset = "davis"
    model_name = "GINConvNet"
    model_file = f"model_{model_name}_{dataset}.model"

    # Maximum allowed prediction difference between the full ligand and a candidate fragment.
    epsilon = 0.25

    # Maximum size of each connected subgraph generated during subgraph mining.
    max_subgraph_size = 8

    # Maximum number of pairwise-disjoint subgraphs allowed in one candidate explanation.
    max_combination_size = 2

    # Maximum total number of atoms in the union of all subgraphs in one candidate explanation.
    max_total_atoms = 8

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Could not find model file: {model_file}")

    device = get_device()
    print("device:", device)

    test_data = TestbedDataset(root="data", dataset=dataset + "_test")

    # Index of the test sample used for the prototype run.
    sample_index = 948
    full_data = test_data[sample_index]

    protein_feat_dim = full_data.protein_feat.view(-1).shape[0]
    model = GINConvNet(protein_feat_dim=protein_feat_dim).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    df = pd.read_csv("data/" + dataset + "_test.csv")
    smiles = df.iloc[sample_index]["compound_iso_smiles"]

    print("SMILES:", smiles)

    mining_graph = smile_to_mining_graph(smiles)

    print("atom_count:", mining_graph["atom_count"])

    subgraph_count = sum(
        1
        for _ in enumerate_connected_subgraphs(
            mining_graph["neighbor_map"],
            max_size=max_subgraph_size
        )
    )
    print("number of connected subgraphs:", subgraph_count)

    full_prediction = predict_single(model, device, full_data)

    subgraphs = enumerate_connected_subgraphs(
        mining_graph["neighbor_map"],
        max_size=max_subgraph_size
    )

    candidate_combinations = generate_pairwise_disjoint_combinations(
        subgraphs,
        max_combination_size=max_combination_size,
        max_total_atoms=max_total_atoms
    )

    print("Predicting candidates in batches...")

    sufficient_subgraphs = []
    candidate_count = 0

    for record, subgraph_prediction in predict_candidate_batches(
        model,
        device,
        full_data,
        mining_graph,
        candidate_combinations,
        batch_size=256
    ):
        candidate_count += 1

        if candidate_count == 1 or candidate_count % 5000 == 0:
            print(
                f"Processed candidates: {candidate_count}",
                flush=True
            )

        difference = abs(full_prediction - subgraph_prediction)

        if difference <= epsilon:
            sufficient_subgraphs.append({
                **record,
                "prediction": subgraph_prediction,
                "difference": difference,
            })

    print("number of candidate combinations:", candidate_count)

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

    output_file = "minimal_sufficient_subgraphs.csv"

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "sample_index",
            "full_smiles",
            "fragment_smiles",
            "atom_indices",
            "num_components",
            "components",
            "size",
            "full_prediction",
            "fragment_prediction",
            "difference",
            "epsilon",
            "max_subgraph_size",
            "dataset",
            "model_name",
            "max_combination_size",
            "max_total_atoms",
            "component_smiles",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in minimal_sufficient_subgraphs:
            fragment_smiles = subgraph_to_smiles(
                mining_graph,
                item["mask"]
            )

            components = item.get("components", (item["mask"],))

            component_smiles = [
                subgraph_to_smiles(mining_graph, component)
                for component in components
            ]

            writer.writerow({
                "dataset": dataset,
                "sample_index": sample_index,
                "model_name": model_name,
                "full_smiles": smiles,
                "fragment_smiles": fragment_smiles,
                "atom_indices": bitmask_to_atoms(item["mask"]),
                "num_components": item["num_components"],
                "components": json.dumps(format_subgraph_components(components)),
                "component_smiles": json.dumps(component_smiles),
                "size": item["size"],
                "full_prediction": full_prediction,
                "fragment_prediction": item["prediction"],
                "difference": item["difference"],
                "epsilon": epsilon,
                "max_subgraph_size": max_subgraph_size,
                "max_combination_size": max_combination_size,
                "max_total_atoms": max_total_atoms,
            })

    print(f"\nSaved results to {output_file}")

    print("\nFull prediction:")
    print(full_prediction)

    print(f"\nSufficient subgraphs with epsilon={epsilon}:")
    print("count:", len(sufficient_subgraphs))

    print(f"\nMinimal sufficient subgraphs with epsilon={epsilon}:")
    print("count:", len(minimal_sufficient_subgraphs))

    print("\nTop minimal sufficient subgraphs:")
    for item in minimal_sufficient_subgraphs[:20]:
        atom_descriptions = describe_subgraph_atoms(
            mining_graph,
            item["mask"]
        )

        fragment_smiles = subgraph_to_smiles(
            mining_graph,
            item["mask"]
        )

        print(
            "atoms:", bitmask_to_atoms(item["mask"]),
            "| components:", item["num_components"],
            "| size:", item["size"],
            "| prediction:", round(item["prediction"], 4),
            "| difference:", round(item["difference"], 4),
            "| fragment SMILES:", fragment_smiles
        )

        print("atom details:")
        for atom_info in atom_descriptions:
            print(
                "  idx:", atom_info["atom_idx"],
                "| symbol:", atom_info["symbol"],
                "| aromatic:", atom_info["is_aromatic"],
                "| ring:", atom_info["is_in_ring"]
            )


if __name__ == "__main__":
    main()