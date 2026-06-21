import csv
import torch
from torch_geometric.loader import DataLoader


class GradientSaliencyExplainer:
    """
    Gradient-based ligand atom importance for GraphDTA-style PyG models.

    Supports two saliency variants:
      1. node_mask saliency: gradient of prediction with respect to an all-one node mask
      2. x saliency: gradient of prediction with respect to ligand atom feature vectors

    The node_mask variant fits models that multiply x by data.node_mask in forward().
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _single_batch(self, full_data):
        loader = DataLoader([full_data], batch_size=1, shuffle=False)
        for batch in loader:
            return batch.to(self.device)
        raise ValueError("Could not create batch from full_data.")

    def _atom_symbol(self, mining_graph, atom_idx):
        mol = mining_graph.get("mol") if mining_graph is not None else None
        if mol is None:
            return None
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()

    def explain_node_mask(self, full_data, mining_graph=None):
        """
        Score atoms by |d prediction / d node_mask_i|.

        This is usually the best fit for your PI setup because PI explanations
        are also represented by node masks.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        batch = self._single_batch(full_data)

        # Create a full-ligand mask and make it differentiable.
        node_mask = torch.ones(
            (batch.x.size(0), 1),
            dtype=batch.x.dtype,
            device=batch.x.device,
            requires_grad=True,
        )
        batch.node_mask = node_mask

        prediction = self.model(batch)
        scalar_prediction = prediction.view(-1).sum()
        scalar_prediction.backward()

        if node_mask.grad is None:
            raise RuntimeError(
                "node_mask.grad is None. The model may not use data.node_mask "
                "in a differentiable way. Try explain_x(...) instead."
            )

        gradients = node_mask.grad.detach().view(-1).cpu()
        importances = gradients.abs()
        prediction_value = prediction.detach().cpu().view(-1)[0].item()

        results = []
        for atom_idx, (gradient, importance) in enumerate(zip(gradients, importances)):
            results.append({
                "atom_idx": atom_idx,
                "atom_symbol": self._atom_symbol(mining_graph, atom_idx),
                "prediction": prediction_value,
                "signed_gradient": float(gradient.item()),
                "importance": float(importance.item()),
                "method": "gradient_node_mask",
            })

        return sorted(results, key=lambda item: item["importance"], reverse=True)

    def explain_x(self, full_data, mining_graph=None):
        """
        Score atoms by sum_j |d prediction / d x_ij| over atom features.

        This is a fallback if node_mask gradients do not work.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        batch = self._single_batch(full_data)

        # Make ligand atom features differentiable.
        batch.x = batch.x.detach().clone().requires_grad_(True)

        # Use a full-ligand mask if the model expects/uses node_mask.
        batch.node_mask = torch.ones(
            (batch.x.size(0), 1),
            dtype=batch.x.dtype,
            device=batch.x.device,
        )

        prediction = self.model(batch)
        scalar_prediction = prediction.view(-1).sum()
        scalar_prediction.backward()

        if batch.x.grad is None:
            raise RuntimeError("x.grad is None. Could not compute gradients w.r.t. x.")

        gradients = batch.x.grad.detach().cpu()
        importances = gradients.abs().sum(dim=1)
        signed_scores = gradients.sum(dim=1)
        prediction_value = prediction.detach().cpu().view(-1)[0].item()

        results = []
        for atom_idx, (signed_score, importance) in enumerate(zip(signed_scores, importances)):
            results.append({
                "atom_idx": atom_idx,
                "atom_symbol": self._atom_symbol(mining_graph, atom_idx),
                "prediction": prediction_value,
                "signed_gradient": float(signed_score.item()),
                "importance": float(importance.item()),
                "method": "gradient_x",
            })

        return sorted(results, key=lambda item: item["importance"], reverse=True)

    def save_results_csv(self, results, output_file):
        if not results:
            raise ValueError("No saliency results to save.")

        fieldnames = [
            "atom_idx",
            "atom_symbol",
            "prediction",
            "signed_gradient",
            "importance",
            "method",
        ]

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in results:
                writer.writerow(item)
