import csv
import json

import torch
from torch_geometric.loader import DataLoader

from subgraph_explainer import (
    attach_target_fields,
    bitmask_to_atoms,
    subgraph_to_pyg_data,
    subgraph_to_smiles,
)


class OcclusionExplainer:
    """
    Ligand occlusion baseline for GraphDTA-style PyG samples.

    This class does not physically delete atoms from the graph. It uses the same
    full-graph + node_mask approach as the PI/subgraph pipeline:

        node_mask = 1  -> atom is visible to the model
        node_mask = 0  -> atom is hidden from the model

    """

    def __init__(self, model, device, prediction_batch_size=256):
        self.model = model
        self.device = device
        self.prediction_batch_size = prediction_batch_size

    def predict_single(self, data):
        """Run model prediction for one PyG Data sample."""
        self.model.eval()
        loader = DataLoader([data], batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                prediction = self.model(batch)

        return prediction.cpu().item()

    def predict_many(self, data_list):
        """Run batched model predictions for many PyG Data samples."""
        self.model.eval()
        predictions = []

        loader = DataLoader(
            data_list,
            batch_size=self.prediction_batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                predictions.extend(output.cpu().view(-1).tolist())

        return predictions

    def explain_atoms(
        self,
        full_data,
        mining_graph,
        full_prediction=None,
        sort_by_importance=True,
    ):
        """
        Compute leave-one-atom-out occlusion scores.

        For each atom i:
            1. keep all atoms except i visible
            2. predict again
            3. importance = abs(full_prediction - occluded_prediction)

        Returns a list of dictionaries, one per atom.
        """
        atom_count = mining_graph["atom_count"]
        full_mask = (1 << atom_count) - 1

        if full_prediction is None:
            full_prediction = self.predict_single(full_data)

        occluded_data_list = []
        records = []

        for atom_idx in range(atom_count):
            occluded_mask = full_mask & ~(1 << atom_idx)
            occluded_data = subgraph_to_pyg_data(mining_graph, occluded_mask)
            occluded_data = attach_target_fields(occluded_data, full_data)

            atom = mining_graph["mol"].GetAtomWithIdx(atom_idx)

            occluded_data_list.append(occluded_data)
            records.append({
                "atom_idx": atom_idx,
                "atom_symbol": atom.GetSymbol(),
                "is_aromatic": atom.GetIsAromatic(),
                "is_in_ring": atom.IsInRing(),
                "visible_atom_indices": bitmask_to_atoms(occluded_mask),
                "occluded_mask": occluded_mask,
            })

        occluded_predictions = self.predict_many(occluded_data_list)

        results = []
        for record, occluded_prediction in zip(records, occluded_predictions):
            prediction_delta = full_prediction - occluded_prediction
            importance = abs(prediction_delta)

            results.append({
                **record,
                "full_prediction": full_prediction,
                "occluded_prediction": occluded_prediction,
                "prediction_delta": prediction_delta,
                "importance": importance,
            })

        if sort_by_importance:
            results.sort(key=lambda item: item["importance"], reverse=True)

        return results

    def explain_fragments(
        self,
        full_data,
        mining_graph,
        fragment_masks,
        full_prediction=None,
        sort_by_importance=True,
    ):
        """
        Occlude whole PI fragments or candidate fragments.

        This is useful after PI has found fragments. For each fragment mask:
            1. hide all atoms in the fragment
            2. keep all other atoms visible
            3. measure how strongly the prediction changes

        Returns a list of dictionaries, one per fragment.
        """
        atom_count = mining_graph["atom_count"]
        full_mask = (1 << atom_count) - 1

        if full_prediction is None:
            full_prediction = self.predict_single(full_data)

        occluded_data_list = []
        records = []

        for fragment_index, fragment_mask in enumerate(fragment_masks):
            occluded_mask = full_mask & ~fragment_mask
            fragment_atoms = bitmask_to_atoms(fragment_mask)

            occluded_data = subgraph_to_pyg_data(mining_graph, occluded_mask)
            occluded_data = attach_target_fields(occluded_data, full_data)

            occluded_data_list.append(occluded_data)
            records.append({
                "fragment_index": fragment_index,
                "fragment_mask": fragment_mask,
                "fragment_atom_indices": fragment_atoms,
                "fragment_size": len(fragment_atoms),
                "fragment_smiles": subgraph_to_smiles(mining_graph, fragment_mask),
                "visible_atom_indices": bitmask_to_atoms(occluded_mask),
                "occluded_mask": occluded_mask,
            })

        occluded_predictions = self.predict_many(occluded_data_list)

        results = []
        for record, occluded_prediction in zip(records, occluded_predictions):
            prediction_delta = full_prediction - occluded_prediction
            importance = abs(prediction_delta)

            results.append({
                **record,
                "full_prediction": full_prediction,
                "occluded_prediction": occluded_prediction,
                "prediction_delta": prediction_delta,
                "importance": importance,
            })

        if sort_by_importance:
            results.sort(key=lambda item: item["importance"], reverse=True)

        return results

    @staticmethod
    def save_atom_results_csv(results, output_file):
        """Save leave-one-atom-out occlusion results to CSV."""
        fieldnames = [
            "atom_idx",
            "atom_symbol",
            "is_aromatic",
            "is_in_ring",
            "full_prediction",
            "occluded_prediction",
            "prediction_delta",
            "importance",
            "visible_atom_indices",
        ]

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for item in results:
                writer.writerow({
                    "atom_idx": item["atom_idx"],
                    "atom_symbol": item["atom_symbol"],
                    "is_aromatic": item["is_aromatic"],
                    "is_in_ring": item["is_in_ring"],
                    "full_prediction": item["full_prediction"],
                    "occluded_prediction": item["occluded_prediction"],
                    "prediction_delta": item["prediction_delta"],
                    "importance": item["importance"],
                    "visible_atom_indices": json.dumps(item["visible_atom_indices"]),
                })

    @staticmethod
    def save_fragment_results_csv(results, output_file):
        """Save fragment occlusion results to CSV."""
        fieldnames = [
            "fragment_index",
            "fragment_smiles",
            "fragment_atom_indices",
            "fragment_size",
            "full_prediction",
            "occluded_prediction",
            "prediction_delta",
            "importance",
            "visible_atom_indices",
        ]

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for item in results:
                writer.writerow({
                    "fragment_index": item["fragment_index"],
                    "fragment_smiles": item["fragment_smiles"],
                    "fragment_atom_indices": json.dumps(item["fragment_atom_indices"]),
                    "fragment_size": item["fragment_size"],
                    "full_prediction": item["full_prediction"],
                    "occluded_prediction": item["occluded_prediction"],
                    "prediction_delta": item["prediction_delta"],
                    "importance": item["importance"],
                    "visible_atom_indices": json.dumps(item["visible_atom_indices"]),
                })
