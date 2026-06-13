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
    subgraph_to_pyg_data,
    attach_target_fields,
    filter_minimal_sufficient_subgraphs,
    describe_subgraph_atoms,
    subgraph_to_smiles,
    generate_pairwise_disjoint_combinations,
    union_subgraph_combination,
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

    subgraphs = enumerate_connected_subgraphs(
        mining_graph["neighbor_map"],
        max_size=max_subgraph_size
    )

    print("atom_count:", mining_graph["atom_count"])
    print("number of connected subgraphs:", len(subgraphs))

    full_prediction = predict_single(model, device, full_data)

    candidate_combinations = generate_pairwise_disjoint_combinations(
        subgraphs,
        max_combination_size=max_combination_size,
        max_total_atoms=max_total_atoms
    )

    print("number of candidate combinations:", len(candidate_combinations))

    candidate_data_list = []
    candidate_records = []

    print("Building candidate Data objects...")
    total_candidates = len(candidate_combinations)

    for index, subgraph_combination in enumerate(candidate_combinations, start=1):
        if index == 1 or index % 5000 == 0 or index == total_candidates:
            print(
                f"Building candidates: {index}/{total_candidates}",
                flush=True
            )

        union_atoms = union_subgraph_combination(subgraph_combination)

        subgraph_data = subgraph_to_pyg_data(mining_graph, union_atoms)
        subgraph_data = attach_target_fields(subgraph_data, full_data)

        candidate_data_list.append(subgraph_data)

        candidate_records.append({
            "atoms": union_atoms,
            "components": subgraph_combination,
            "num_components": len(subgraph_combination),
            "size": len(union_atoms),
        })

    print("Predicting candidates in batches...")

    candidate_predictions = predict_many(
        model,
        device,
        candidate_data_list,
        batch_size=256
    )

    sufficient_subgraphs = []

    for record, subgraph_prediction in zip(candidate_records, candidate_predictions):
        difference = abs(full_prediction - subgraph_prediction)

        if difference <= epsilon:
            sufficient_subgraphs.append({
                **record,
                "prediction": subgraph_prediction,
                "difference": difference,
            })

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
                item["atoms"]
            )

            components = item.get("components", (item["atoms"],))

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
                "atom_indices": sorted(item["atoms"]),
                "num_components": item.get("num_components", 1),
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
            item["atoms"]
        )

        fragment_smiles = subgraph_to_smiles(
            mining_graph,
            item["atoms"]
        )

        print(
            "atoms:", item["atoms"],
            "| components:", item.get("num_components", 1),
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