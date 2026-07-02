import argparse
import csv
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.loader import DataLoader

from dataset import TestbedDataset
from experiment_pi_fidelity import (
    build_masked_candidate_data,
    compute_indispensability,
    get_device,
    load_model,
    pick_best_candidate,
    predict_many,
    predict_single,
    prepare_data_for_model,
)
from kinase_features import build_all_target_feature_tables
from ligand_features import atom_features, molecule_features, pharmacophore_atom_flags
from models.ginconv import GINConvNet
from subgraph_explainer import (
    attach_target_fields,
    bitmask_to_atoms,
    deduplicate_sufficient_subgraphs_by_atoms,
    enumerate_connected_subgraphs,
    filter_minimal_sufficient_subgraphs,
    format_subgraph_components,
    generate_pairwise_disjoint_combinations,
    smile_to_mining_graph,
    subgraph_to_smiles,
)
from utils import ci, mse, pearson, rmse, spearman


SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}
MAX_SEQ_LEN = 1000


def seq_cat(prot):
    x = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(str(prot)[:MAX_SEQ_LEN]):
        x[i] = SEQ_DICT.get(ch, 0)
    return x


def seq_cat_unk(prot, max_len):
    x = np.zeros(max_len)
    for i, ch in enumerate(str(prot)[:max_len]):
        x[i] = SEQ_DICT.get(ch, 0)
    return x


def one_hot_value(value, vocab):
    return [1.0 if value == v else 0.0 for v in vocab]


def build_target_feature_map(feature_df):
    feature_df = feature_df.fillna("")

    group_vocab = sorted(feature_df["kinase_group"].unique())
    family_vocab = sorted(feature_df["kinase_family"].unique())
    subfamily_vocab = sorted(feature_df["kinase_subfamily"].unique())
    dfg_vocab = sorted(feature_df["dfg_state"].unique())
    ac_vocab = sorted(feature_df["ac_helix_state"].unique())

    target_feat_map = {}

    for _, row in feature_df.iterrows():
        feat = []
        feat += [float(row["has_structure"]) if str(row["has_structure"]).strip() != "" else 0.0]
        feat += [float(row["pocket_hydrophobicity"]) if str(row["pocket_hydrophobicity"]).strip() != "" else 0.0]
        feat += [float(row["pocket_charge"]) if str(row["pocket_charge"]).strip() != "" else 0.0]
        feat += [float(row["activation_loop_state"]) if str(row["activation_loop_state"]).strip() != "" else 0.0]
        feat += one_hot_value(row["kinase_group"], group_vocab)
        feat += one_hot_value(row["kinase_family"], family_vocab)
        feat += one_hot_value(row["kinase_subfamily"], subfamily_vocab)
        feat += one_hot_value(row["dfg_state"], dfg_vocab)
        feat += one_hot_value(row["ac_helix_state"], ac_vocab)
        target_feat_map[row["target_id"]] = np.asarray(feat, dtype=np.float32)

    return target_feat_map


def build_target_sequence_maps(feature_df):
    feature_df = feature_df.fillna("")
    pocket_map = {}
    gatekeeper_map = {}
    hinge_map = {}

    for _, row in feature_df.iterrows():
        target_id = row["target_id"]
        pocket_map[target_id] = seq_cat_unk(row["klifs_85_residue_sequence"], 85)
        gatekeeper_map[target_id] = seq_cat_unk(row["gatekeeper_residue"], 1)
        hinge_map[target_id] = seq_cat_unk(row["hinge_residues"], 3)

    return pocket_map, gatekeeper_map, hinge_map


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_features = molecule_features(mol)
    pharm_flags = pharmacophore_atom_flags(mol)

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom, pharm_flags[atom.GetIdx()])
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    graph = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in graph.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index, mol_features


def build_smile_graph(smiles_list):
    smile_graph = {}
    for smile in sorted(set(smiles_list)):
        smile_graph[smile] = smile_to_graph(smile)
    return smile_graph


def predict_candidate_batches_quiet(
    model,
    device,
    full_data,
    mining_graph,
    candidate_iter,
    protein_feat_dim,
    candidate_buffer_size=256,
    prediction_batch_size=256,
):
    model.eval()
    candidate_data_buffer = []
    candidate_record_buffer = []

    def flush():
        nonlocal candidate_data_buffer
        nonlocal candidate_record_buffer
        if not candidate_data_buffer:
            return

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


def explain_sample_quiet(
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

    for record, subgraph_prediction in predict_candidate_batches_quiet(
        model,
        device,
        full_data,
        mining_graph,
        candidate_combinations,
        protein_feat_dim,
        candidate_buffer_size=candidate_buffer_size,
        prediction_batch_size=prediction_batch_size,
    ):
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
        key=lambda item: (item["size"], item["difference"]),
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

    best_candidate = pick_best_candidate(minimal_sufficient_subgraphs)
    return full_prediction, minimal_sufficient_subgraphs, best_candidate, mining_graph


def generate_pi_csv(
    dataset,
    split,
    output_path,
    epsilon,
    max_subgraph_size,
    max_combination_size,
    max_total_atoms,
    candidate_buffer_size,
    prediction_batch_size,
    start_index,
    end_index,
    fallback_original,
):
    device = get_device()
    csv_path = f"data/{dataset}_{split}.csv"
    df = pd.read_csv(csv_path).copy()
    split_data = TestbedDataset(root="data", dataset=f"{dataset}_{split}")

    actual_end = len(df) if end_index is None else min(end_index, len(df))
    if start_index < 0 or start_index >= actual_end:
        raise ValueError("Invalid start/end index range.")

    protein_feat_dim = split_data[0].protein_feat.view(-1).shape[0]
    model, protein_feat_dim = load_model(
        "GINConvNet",
        dataset,
        device,
        protein_feat_dim,
    )

    output_rows = []
    print(f"Generating PI-only CSV for {split} on rows {start_index}:{actual_end}")

    for sample_index in range(len(df)):
        row = df.iloc[sample_index].to_dict()
        original_smiles = row["compound_iso_smiles"]
        row["original_compound_iso_smiles"] = original_smiles

        if sample_index < start_index or sample_index >= actual_end:
            row["pi_found"] = None
            row["pi_fragment_smiles"] = None
            row["pi_atom_indices"] = None
            row["pi_components"] = None
            row["pi_component_smiles"] = None
            row["pi_num_components"] = None
            row["pi_size"] = None
            row["pi_prediction"] = None
            row["pi_full_prediction_difference"] = None
            row["pi_indispensability"] = None
            output_rows.append(row)
            continue

        if sample_index == start_index or (sample_index - start_index) % 10 == 0:
            print(f"  {split}: sample {sample_index}/{actual_end - 1}", flush=True)

        full_data = split_data[sample_index]
        full_prediction, _, best_candidate, mining_graph = explain_sample_quiet(
            model=model,
            device=device,
            full_data=full_data,
            protein_feat_dim=protein_feat_dim,
            smiles=original_smiles,
            epsilon=epsilon,
            max_subgraph_size=max_subgraph_size,
            max_combination_size=max_combination_size,
            max_total_atoms=max_total_atoms,
            candidate_buffer_size=candidate_buffer_size,
            prediction_batch_size=prediction_batch_size,
        )

        row["full_prediction"] = full_prediction

        if best_candidate is None:
            row["pi_found"] = False
            row["pi_fragment_smiles"] = None
            row["pi_atom_indices"] = None
            row["pi_components"] = None
            row["pi_component_smiles"] = None
            row["pi_num_components"] = None
            row["pi_size"] = None
            row["pi_prediction"] = None
            row["pi_full_prediction_difference"] = None
            row["pi_indispensability"] = None
            if fallback_original:
                row["compound_iso_smiles"] = original_smiles
        else:
            components = best_candidate.get("components", (best_candidate["mask"],))
            row["pi_found"] = True
            row["pi_fragment_smiles"] = subgraph_to_smiles(mining_graph, best_candidate["mask"])
            row["pi_atom_indices"] = json.dumps(bitmask_to_atoms(best_candidate["mask"]))
            row["pi_components"] = json.dumps(format_subgraph_components(components))
            row["pi_component_smiles"] = json.dumps([
                subgraph_to_smiles(mining_graph, component)
                for component in components
            ])
            row["pi_num_components"] = best_candidate["num_components"]
            row["pi_size"] = best_candidate["size"]
            row["pi_prediction"] = best_candidate["prediction"]
            row["pi_full_prediction_difference"] = best_candidate["difference"]
            row["pi_indispensability"] = best_candidate["indispensability"]
            row["compound_iso_smiles"] = row["pi_fragment_smiles"]

        output_rows.append(row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False)
    print(f"Saved {split} PI-only CSV to {output_path}")


def build_processed_dataset(dataset, split, csv_path, processed_name, force_rebuild=False):
    build_all_target_feature_tables()

    processed_file = f"data/processed/{processed_name}.pt"
    if os.path.isfile(processed_file) and not force_rebuild:
        print(f"{processed_file} already exists, skipping build")
        return

    target_feature_df = pd.read_csv(f"data/{dataset}_target_features.csv")
    target_feature_map = build_target_feature_map(target_feature_df)
    pocket_map, gatekeeper_map, hinge_map = build_target_sequence_maps(target_feature_df)

    df = pd.read_csv(csv_path)
    drugs = list(df["compound_iso_smiles"])
    target_ids = list(df["target_id"])
    prots = list(df["target_sequence"])
    labels = list(df["affinity"])

    xt = [seq_cat(t) for t in prots]
    xk = [pocket_map[tid] for tid in target_ids]
    xg = [gatekeeper_map[tid] for tid in target_ids]
    xh = [hinge_map[tid] for tid in target_ids]
    xp = [target_feature_map[tid] for tid in target_ids]

    smile_graph = build_smile_graph(drugs)

    print(f"Building processed dataset {processed_name} from {csv_path}")
    TestbedDataset(
        root="data",
        dataset=processed_name,
        xd=np.asarray(drugs),
        xt=np.asarray(xt),
        xp=np.asarray(xp, dtype=np.float32),
        xk=np.asarray(xk),
        xg=np.asarray(xg),
        xh=np.asarray(xh),
        y=np.asarray(labels),
        smile_graph=smile_graph,
    )


def train_epoch(model, device, train_loader, optimizer, loss_fn, epoch, log_interval):
    print(f"Training on {len(train_loader.dataset)} samples...")
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data.x),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def predict_dataset(model, device, loader):
    model.eval()
    total_preds = []
    total_labels = []
    print(f"Make prediction for {len(loader.dataset)} samples...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds.extend(output.cpu().view(-1).tolist())
            total_labels.extend(data.y.view(-1).cpu().tolist())
    return np.asarray(total_labels), np.asarray(total_preds)


def train_pi_model(
    dataset,
    train_processed_name,
    test_processed_name,
    epochs,
    train_batch_size,
    test_batch_size,
    lr,
    eval_interval,
    log_interval,
    split_seed,
):
    device = get_device()
    print("device:", device)

    train_data_full = TestbedDataset(root="data", dataset=train_processed_name)
    test_data = TestbedDataset(root="data", dataset=test_processed_name)

    protein_feat_dim = train_data_full[0].protein_feat.view(-1).shape[0]
    print("protein_feat_dim:", protein_feat_dim)

    train_size = int(0.8 * len(train_data_full))
    valid_size = len(train_data_full) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_data, valid_data = torch.utils.data.random_split(
        train_data_full,
        [train_size, valid_size],
        generator=generator,
    )

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    model = GINConvNet(protein_feat_dim=protein_feat_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_mse = float("inf")
    best_result = None
    best_epoch = -1

    model_file_name = f"model_GINConvNet_{dataset}_pi_only.model"
    result_file_name = f"result_GINConvNet_{dataset}_pi_only.csv"
    comparison_file_name = f"comparison_GINConvNet_{dataset}_pi_only.csv"

    for epoch in range(epochs):
        train_epoch(model, device, train_loader, optimizer, loss_fn, epoch + 1, log_interval)
        if ((epoch + 1) % eval_interval != 0) and (epoch != epochs - 1):
            continue

        print("predicting for valid data")
        valid_g, valid_p = predict_dataset(model, device, valid_loader)
        valid_mse = mse(valid_g, valid_p)

        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)

            print("predicting for test data")
            test_g, test_p = predict_dataset(model, device, test_loader)
            best_result = [
                rmse(test_g, test_p),
                mse(test_g, test_p),
                pearson(test_g, test_p),
                spearman(test_g, test_p),
                ci(test_g, test_p),
            ]

            with open(result_file_name, "w") as handle:
                handle.write(",".join(map(str, best_result)))

            print(
                "validation mse improved at epoch",
                best_epoch,
                "; best_valid_mse:",
                best_valid_mse,
                "; best_test_mse,best_test_ci:",
                best_result[1],
                best_result[-1],
            )
        else:
            print(
                valid_mse,
                "No improvement since epoch",
                best_epoch,
                "; best_valid_mse:",
                best_valid_mse,
            )

    baseline_result_file = f"result_GINConvNet_{dataset}.csv"
    comparison_rows = [
        {
            "setting": "pi_only",
            "rmse": best_result[0],
            "mse": best_result[1],
            "pearson": best_result[2],
            "spearman": best_result[3],
            "ci": best_result[4],
            "best_epoch": best_epoch,
            "best_valid_mse": best_valid_mse,
        }
    ]

    if os.path.isfile(baseline_result_file):
        values = [float(value) for value in open(baseline_result_file).read().strip().split(",")]
        comparison_rows.append({
            "setting": "full_model_baseline",
            "rmse": values[0],
            "mse": values[1],
            "pearson": values[2],
            "spearman": values[3],
            "ci": values[4],
            "best_epoch": None,
            "best_valid_mse": None,
        })

    with open(comparison_file_name, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting",
                "rmse",
                "mse",
                "pearson",
                "spearman",
                "ci",
                "best_epoch",
                "best_valid_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Saved PI-only metrics to {result_file_name}")
    print(f"Saved PI-only comparison to {comparison_file_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2 experiment: generate PI-only CSVs and retrain GINConvNet on them."
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "build", "train", "all"],
        default="all",
    )
    parser.add_argument(
        "--dataset",
        default="davis",
        choices=["davis", "kiba"],
    )
    parser.add_argument(
        "--train-pi-csv",
        default=None,
    )
    parser.add_argument(
        "--test-pi-csv",
        default=None,
    )
    parser.add_argument(
        "--train-processed-name",
        default=None,
    )
    parser.add_argument(
        "--test-processed-name",
        default=None,
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
        default=1024,
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--train-start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train-end-index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--test-start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--test-end-index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fallback-original",
        action="store_true",
        help="Use the original full-molecule SMILES when no PI is found.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_pi_csv = args.train_pi_csv or f"data/{args.dataset}_train_pi_gin.csv"
    test_pi_csv = args.test_pi_csv or f"data/{args.dataset}_test_pi_gin.csv"
    train_processed_name = args.train_processed_name or f"{args.dataset}_pi_gin_train"
    test_processed_name = args.test_processed_name or f"{args.dataset}_pi_gin_test"

    if args.mode in ("generate", "all"):
        generate_pi_csv(
            dataset=args.dataset,
            split="train",
            output_path=train_pi_csv,
            epsilon=args.epsilon,
            max_subgraph_size=args.max_subgraph_size,
            max_combination_size=args.max_combination_size,
            max_total_atoms=args.max_total_atoms,
            candidate_buffer_size=args.candidate_buffer_size,
            prediction_batch_size=args.prediction_batch_size,
            start_index=args.train_start_index,
            end_index=args.train_end_index,
            fallback_original=args.fallback_original,
        )
        generate_pi_csv(
            dataset=args.dataset,
            split="test",
            output_path=test_pi_csv,
            epsilon=args.epsilon,
            max_subgraph_size=args.max_subgraph_size,
            max_combination_size=args.max_combination_size,
            max_total_atoms=args.max_total_atoms,
            candidate_buffer_size=args.candidate_buffer_size,
            prediction_batch_size=args.prediction_batch_size,
            start_index=args.test_start_index,
            end_index=args.test_end_index,
            fallback_original=args.fallback_original,
        )

    if args.mode in ("build", "all"):
        build_processed_dataset(
            args.dataset,
            "train",
            train_pi_csv,
            train_processed_name,
            force_rebuild=args.force_rebuild,
        )
        build_processed_dataset(
            args.dataset,
            "test",
            test_pi_csv,
            test_processed_name,
            force_rebuild=args.force_rebuild,
        )

    if args.mode in ("train", "all"):
        train_pi_model(
            dataset=args.dataset,
            train_processed_name=train_processed_name,
            test_processed_name=test_processed_name,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            lr=args.lr,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            split_seed=args.split_seed,
        )


if __name__ == "__main__":
    main()
