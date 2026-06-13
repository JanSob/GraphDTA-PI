import torch
from rdkit import Chem
import numpy as np
from itertools import combinations
from torch_geometric import data as DATA

from ligand_features import (
    molecule_features,
    pharmacophore_atom_flags,
    atom_features,
)

"""
Subgraph explanation utilities.

Builds ligand mining graphs, enumerates connected subgraphs, converts
selected atom sets into masked full-graph PyG inputs, and handles
sufficient fragment combinations.
"""

def smile_to_mining_graph(smile):
    """Convert a SMILES string into a ligand graph used for subgraph mining."""
    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    molecule_level_features = molecule_features(mol)
    pharmacophore_flags = pharmacophore_atom_flags(mol)

    atom_indices = []
    atom_features_by_index = {}
    neighbor_map = {}
    undirected_edges = []

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()

        atom_indices.append(atom_idx)
        neighbor_map[atom_idx] = []

        feature = atom_features(atom, pharmacophore_flags[atom_idx])
        atom_features_by_index[atom_idx] = feature / sum(feature)

    for bond in mol.GetBonds():
        atom_a_idx = bond.GetBeginAtomIdx()
        atom_b_idx = bond.GetEndAtomIdx()

        undirected_edges.append((atom_a_idx, atom_b_idx))

        neighbor_map[atom_a_idx].append(atom_b_idx)
        neighbor_map[atom_b_idx].append(atom_a_idx)

    return {
        "mol": mol,
        "atom_indices": atom_indices,
        "undirected_edges": undirected_edges,
        "neighbor_map": neighbor_map,
        "atom_features_by_index": atom_features_by_index,
        "molecule_level_features": molecule_level_features,
        "atom_count": mol.GetNumAtoms(),
    }


def enumerate_connected_subgraphs(neighbor_map, max_size=None):
    """Enumerate connected induced subgraphs up to an optional atom limit."""

    def compute_distances_from_anchor(current_atoms, anchor):
        distances = {anchor: 0}
        queue = [anchor]

        while queue:
            current_atom = queue.pop(0)

            for neighbor_atom in neighbor_map[current_atom]:
                if neighbor_atom in current_atoms and neighbor_atom not in distances:
                    distances[neighbor_atom] = distances[current_atom] + 1
                    queue.append(neighbor_atom)

        return distances

    def candidate_distance(candidate_atom, current_atoms, distances):
        possible_distances = []

        for neighbor_atom in neighbor_map[candidate_atom]:
            if neighbor_atom in current_atoms:
                possible_distances.append(distances[neighbor_atom] + 1)

        if len(possible_distances) == 0:
            return None

        return min(possible_distances)

    def get_utmost_atom(current_atoms, distances):
        return max(
            current_atoms,
            key=lambda atom_idx: (distances[atom_idx], atom_idx)
        )

    def is_valid_extension(current_atoms, candidate_atom):
        anchor = min(current_atoms)

        if candidate_atom < anchor:
            return False

        distances = compute_distances_from_anchor(current_atoms, anchor)
        utmost_atom = get_utmost_atom(current_atoms, distances)

        dist_candidate = candidate_distance(candidate_atom, current_atoms, distances)
        dist_utmost = distances[utmost_atom]

        if dist_candidate is None:
            return False

        if dist_candidate > dist_utmost:
            return True

        return dist_candidate == dist_utmost and candidate_atom > utmost_atom

    def expand(current_atoms, candidate_atoms):
        connected_subgraphs.append(set(current_atoms))

        if max_size is not None and len(current_atoms) >= max_size:
            return

        for candidate_atom in sorted(candidate_atoms):
            if not is_valid_extension(current_atoms, candidate_atom):
                continue

            new_atoms = set(current_atoms)
            new_atoms.add(candidate_atom)

            new_candidates = set(candidate_atoms)
            new_candidates.discard(candidate_atom)

            for neighbor_atom in neighbor_map[candidate_atom]:
                if neighbor_atom not in new_atoms:
                    new_candidates.add(neighbor_atom)

            expand(new_atoms, new_candidates)

    connected_subgraphs = []

    for start_atom in sorted(neighbor_map.keys()):
        expand(
            current_atoms={start_atom},
            candidate_atoms=set(neighbor_map[start_atom])
        )

    return connected_subgraphs


def subgraph_to_pyg_data(mining_graph, subgraph_atoms):
    """Convert selected atoms into a masked full-graph PyTorch Geometric sample."""
    selected_atoms = set(subgraph_atoms)
    full_atom_indices = sorted(mining_graph["atom_indices"])

    features = []
    node_mask = []

    for atom_idx in full_atom_indices:
        atom_feature = mining_graph["atom_features_by_index"][atom_idx]

        if atom_idx in selected_atoms:
            features.append(atom_feature)
            node_mask.append([1.0])
        else:
            features.append(np.zeros_like(atom_feature, dtype=np.float32))
            node_mask.append([0.0])

    edge_index = []
    for atom_a_idx, atom_b_idx in mining_graph["undirected_edges"]:
        edge_index.append([atom_a_idx, atom_b_idx])
        edge_index.append([atom_b_idx, atom_a_idx])

    if len(edge_index) == 0:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_tensor = torch.LongTensor(edge_index).transpose(1, 0)

    graph_data = DATA.Data(
        x=torch.FloatTensor(np.asarray(features, dtype=np.float32)),
        edge_index=edge_index_tensor
    )

    graph_data.node_mask = torch.FloatTensor(np.asarray(node_mask, dtype=np.float32))
    graph_data.mol_features = torch.zeros(
        (1, len(mining_graph["molecule_level_features"])),
        dtype=torch.float32
    )
    graph_data.__setitem__(
        "c_size",
        torch.LongTensor([len(full_atom_indices)])
    )

    return graph_data


def attach_target_fields(subgraph_data, full_data):
    """Copy protein and target fields from the full sample to a subgraph sample."""
    subgraph_data.target = full_data.target
    subgraph_data.protein_feat = full_data.protein_feat
    subgraph_data.klifs_pocket = full_data.klifs_pocket
    subgraph_data.gatekeeper = full_data.gatekeeper
    subgraph_data.hinge = full_data.hinge

    if hasattr(full_data, "y"):
        subgraph_data.y = full_data.y

    return subgraph_data
def filter_minimal_sufficient_subgraphs(sufficient_subgraphs):
    """Keep only sufficient subgraphs that have no smaller sufficient subset."""
    minimal_subgraphs = []

    for candidate in sufficient_subgraphs:
        candidate_atoms = set(candidate["atoms"])
        candidate_is_minimal = True

        for other in sufficient_subgraphs:
            other_atoms = set(other["atoms"])

            if len(other_atoms) >= len(candidate_atoms):
                continue

            if other_atoms.issubset(candidate_atoms):
                candidate_is_minimal = False
                break

        if candidate_is_minimal:
            minimal_subgraphs.append(candidate)

    return minimal_subgraphs

def describe_subgraph_atoms(mining_graph, subgraph_atoms):
    """Return atom symbols and simple structural flags for a subgraph."""
    mol = mining_graph["mol"]

    atom_descriptions = []
    for atom_idx in sorted(subgraph_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_descriptions.append({
            "atom_idx": atom_idx,
            "symbol": atom.GetSymbol(),
            "is_aromatic": atom.GetIsAromatic(),
            "is_in_ring": atom.IsInRing(),
        })

    return atom_descriptions

def subgraph_to_smiles(mining_graph, subgraph_atoms):
    """Convert selected atom indices into an RDKit fragment SMILES string."""
    mol = mining_graph["mol"]

    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(subgraph_atoms),
        canonical=True,
        isomericSmiles=True
    )

def are_pairwise_disjoint(subgraph_combination):
    """Check whether subgraphs share no atom indices."""
    used_atoms = set()

    for subgraph in subgraph_combination:
        subgraph_atoms = set(subgraph)

        if used_atoms.intersection(subgraph_atoms):
            return False

        used_atoms.update(subgraph_atoms)

    return True


def union_subgraph_combination(subgraph_combination):
    """Return the union of atom indices from a subgraph combination."""
    union_atoms = set()

    for subgraph in subgraph_combination:
        union_atoms.update(subgraph)

    return union_atoms


def generate_pairwise_disjoint_combinations(subgraphs, max_combination_size=2, max_total_atoms=None):
    """Generate pairwise-disjoint subgraph combinations under the given size limits."""

    candidates = []

    # include single subgraphs
    for subgraph in subgraphs:
        if max_total_atoms is None or len(subgraph) <= max_total_atoms:
            candidates.append((subgraph,))

    # include multi-subgraph combinations
    for combination_size in range(2, max_combination_size + 1):
        for subgraph_combination in combinations(subgraphs, combination_size):
            if not are_pairwise_disjoint(subgraph_combination):
                continue

            union_atoms = union_subgraph_combination(subgraph_combination)

            if max_total_atoms is not None and len(union_atoms) > max_total_atoms:
                continue

            candidates.append(subgraph_combination)

    return candidates

def deduplicate_sufficient_subgraphs_by_atoms(sufficient_subgraphs):
    """Remove duplicate explanations with the same final atom set."""
    best_by_atom_set = {}

    for item in sufficient_subgraphs:
        atom_key = frozenset(item["atoms"])

        if atom_key not in best_by_atom_set:
            best_by_atom_set[atom_key] = item
            continue

        existing = best_by_atom_set[atom_key]

        existing_components = existing.get("num_components", 1)
        current_components = item.get("num_components", 1)

        # Prefer the simpler explanation.
        if current_components < existing_components:
            best_by_atom_set[atom_key] = item
        elif current_components == existing_components and item["difference"] < existing["difference"]:
            best_by_atom_set[atom_key] = item

    return list(best_by_atom_set.values())


def format_subgraph_components(subgraph_combination):
    """Format subgraph components as sorted atom-index lists."""
    return [
        sorted(component)
        for component in subgraph_combination
    ]
