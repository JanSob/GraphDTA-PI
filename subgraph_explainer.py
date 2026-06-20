import torch
from rdkit import Chem
import numpy as np
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

    # cache immutable graph tensors so each candidate only changes the node mask
    adj_list = [tuple(sorted(neighbor_map[i])) for i in range(mol.GetNumAtoms())]

    edge_pairs = []
    for atom_a_idx, atom_b_idx in undirected_edges:
        edge_pairs.append([atom_a_idx, atom_b_idx])
        edge_pairs.append([atom_b_idx, atom_a_idx])

    if edge_pairs:
        edge_index_tensor = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

    full_x_tensor = torch.tensor(
        np.asarray(
            [atom_features_by_index[i] for i in atom_indices],
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )

    return {
        "mol": mol,
        "atom_indices": atom_indices,
        "undirected_edges": undirected_edges,
        "neighbor_map": neighbor_map,
        "adj_list": adj_list,
        "atom_features_by_index": atom_features_by_index,
        "molecule_level_features": molecule_level_features,
        "atom_count": mol.GetNumAtoms(),
        "full_x_tensor": full_x_tensor,
        "edge_index_tensor": edge_index_tensor,
        "mol_features_tensor": torch.zeros(
            (1, len(molecule_level_features)),
            dtype=torch.float32,
        ),
    }

# BITMASK HELPERS
def iter_set_bits(mask):
    """Yield set-bit indices in ascending atom order."""
    while mask:
        least_significant_bit = mask & -mask
        yield least_significant_bit.bit_length() - 1
        mask ^= least_significant_bit


def bitmask_to_atoms(mask):
    """Convert an integer bitmask to sorted atom indices."""
    return list(iter_set_bits(mask))


def atoms_to_bitmask(atoms):
    """Convert atom indices to an integer bitmask."""
    mask = 0

    for atom_idx in atoms:
        mask |= 1 << atom_idx

    return mask
# BITMASK HELPERS END

def iter_connected_subgraph_bitmasks(neighbor_map, max_size=None):
    """Yield connected induced subgraphs as atom bitmasks."""
    vertex_count = len(neighbor_map)
    adjacency = [tuple(sorted(neighbor_map[vertex])) for vertex in range(vertex_count)]
    size_limit = vertex_count if max_size is None else max_size

    # U = current subgraph in visit order
    # C = candidate frontier
    # D = anchor distances
    # P = parent on the anchor path when first inserted into C
    U = []
    C = []
    D = [-1] * vertex_count
    P = [-1] * vertex_count

    # membership arrays keep updates O(1) 
    in_U = [False] * vertex_count
    in_C = [False] * vertex_count
    pos_in_C = [-1] * vertex_count

    current_mask = 0

    def is_valid_child(vertex, anchor, utmost):
        if vertex <= anchor:
            return False

        return (
            D[vertex] > D[utmost]
            or (
                D[vertex] == D[utmost]
                and vertex > utmost
            )
        )

    def depth_first_search(anchor):
        nonlocal current_mask

        yield current_mask

        if len(U) >= size_limit:
            return

        utmost = U[-1]
        candidate_count = len(C)

        for candidate_index in range(candidate_count):
            vertex = C[candidate_index]

            if not in_C[vertex]:
                continue

            if not is_valid_child(vertex, anchor, utmost):
                continue

            removed_index = pos_in_C[vertex]
            moved_vertex = C[-1]

            C[removed_index] = moved_vertex
            pos_in_C[moved_vertex] = removed_index
            C.pop()
            pos_in_C[vertex] = -1
            in_C[vertex] = False
            in_U[vertex] = True
            U.append(vertex)
            current_mask |= 1 << vertex

            child_candidate_base = len(C)

            for neighbor_vertex in adjacency[vertex]:
                if neighbor_vertex <= anchor:
                    continue

                if in_U[neighbor_vertex] or in_C[neighbor_vertex]:
                    continue

                in_C[neighbor_vertex] = True
                D[neighbor_vertex] = D[vertex] + 1
                P[neighbor_vertex] = vertex
                pos_in_C[neighbor_vertex] = len(C)
                C.append(neighbor_vertex)

            yield from depth_first_search(anchor)

            # only remove candidates added by the last child extension
            while len(C) > child_candidate_base:
                neighbor_vertex = C.pop()
                in_C[neighbor_vertex] = False
                pos_in_C[neighbor_vertex] = -1
                D[neighbor_vertex] = -1
                P[neighbor_vertex] = -1

            U.pop()
            in_U[vertex] = False

            C.append(moved_vertex)
            if removed_index < len(C) - 1:
                C[removed_index] = vertex
                pos_in_C[moved_vertex] = len(C) - 1
            else:
                C[removed_index] = vertex

            pos_in_C[vertex] = removed_index
            in_C[vertex] = True
            current_mask ^= 1 << vertex

    for anchor in range(vertex_count):
        U.append(anchor)
        in_U[anchor] = True
        D[anchor] = 0
        P[anchor] = -1
        current_mask = 1 << anchor

        root_candidate_base = len(C)

        for neighbor_vertex in adjacency[anchor]:
            if neighbor_vertex <= anchor:
                continue

            in_C[neighbor_vertex] = True
            D[neighbor_vertex] = 1
            P[neighbor_vertex] = anchor
            pos_in_C[neighbor_vertex] = len(C)
            C.append(neighbor_vertex)

        yield from depth_first_search(anchor)

        while len(C) > root_candidate_base:
            neighbor_vertex = C.pop()
            in_C[neighbor_vertex] = False
            pos_in_C[neighbor_vertex] = -1
            D[neighbor_vertex] = -1
            P[neighbor_vertex] = -1

        U.pop()
        in_U[anchor] = False
        D[anchor] = -1
        P[anchor] = -1
        current_mask = 0


def profile_connected_subgraph_enumerator(neighbor_map, max_size=None):
    """Measure the work terms that correspond to the paper's complexity argument."""
    vertex_count = len(neighbor_map)
    adjacency = [tuple(sorted(neighbor_map[vertex])) for vertex in range(vertex_count)]
    size_limit = vertex_count if max_size is None else max_size

    U = []
    C = []
    D = [-1] * vertex_count
    P = [-1] * vertex_count
    in_U = [False] * vertex_count
    in_C = [False] * vertex_count
    pos_in_C = [-1] * vertex_count

    stats = {
        "vertex_count": vertex_count,
        "output_count": 0,
        "candidate_checks": 0,
        "neighbor_scans": 0,
        "max_depth": 0,
        "max_candidates": 0,
        "max_live_vertices": 0,
        "array_slots": 7 * vertex_count,
    }

    def update_maxima():
        stats["max_depth"] = max(stats["max_depth"], len(U))
        stats["max_candidates"] = max(stats["max_candidates"], len(C))
        stats["max_live_vertices"] = max(stats["max_live_vertices"], len(U) + len(C))

    def is_valid_child(vertex, anchor, utmost):
        stats["candidate_checks"] += 1

        if vertex <= anchor:
            return False

        return (
            D[vertex] > D[utmost]
            or (
                D[vertex] == D[utmost]
                and vertex > utmost
            )
        )

    def depth_first_search(anchor):
        stats["output_count"] += 1
        update_maxima()

        if len(U) >= size_limit:
            return

        utmost = U[-1]
        candidate_count = len(C)

        for candidate_index in range(candidate_count):
            vertex = C[candidate_index]

            if not in_C[vertex]:
                continue

            if not is_valid_child(vertex, anchor, utmost):
                continue

            removed_index = pos_in_C[vertex]
            moved_vertex = C[-1]

            C[removed_index] = moved_vertex
            pos_in_C[moved_vertex] = removed_index
            C.pop()
            pos_in_C[vertex] = -1
            in_C[vertex] = False
            in_U[vertex] = True
            U.append(vertex)
            update_maxima()

            child_candidate_base = len(C)

            for neighbor_vertex in adjacency[vertex]:
                stats["neighbor_scans"] += 1

                if neighbor_vertex <= anchor:
                    continue

                if in_U[neighbor_vertex] or in_C[neighbor_vertex]:
                    continue

                in_C[neighbor_vertex] = True
                D[neighbor_vertex] = D[vertex] + 1
                P[neighbor_vertex] = vertex
                pos_in_C[neighbor_vertex] = len(C)
                C.append(neighbor_vertex)
                update_maxima()

            depth_first_search(anchor)

            while len(C) > child_candidate_base:
                neighbor_vertex = C.pop()
                in_C[neighbor_vertex] = False
                pos_in_C[neighbor_vertex] = -1
                D[neighbor_vertex] = -1
                P[neighbor_vertex] = -1

            U.pop()
            in_U[vertex] = False

            C.append(moved_vertex)
            if removed_index < len(C) - 1:
                C[removed_index] = vertex
                pos_in_C[moved_vertex] = len(C) - 1
            else:
                C[removed_index] = vertex

            pos_in_C[vertex] = removed_index
            in_C[vertex] = True

    for anchor in range(vertex_count):
        U.append(anchor)
        in_U[anchor] = True
        D[anchor] = 0
        P[anchor] = -1
        update_maxima()

        root_candidate_base = len(C)

        for neighbor_vertex in adjacency[anchor]:
            stats["neighbor_scans"] += 1

            if neighbor_vertex <= anchor:
                continue

            in_C[neighbor_vertex] = True
            D[neighbor_vertex] = 1
            P[neighbor_vertex] = anchor
            pos_in_C[neighbor_vertex] = len(C)
            C.append(neighbor_vertex)
            update_maxima()

        depth_first_search(anchor)

        while len(C) > root_candidate_base:
            neighbor_vertex = C.pop()
            in_C[neighbor_vertex] = False
            pos_in_C[neighbor_vertex] = -1
            D[neighbor_vertex] = -1
            P[neighbor_vertex] = -1

        U.pop()
        in_U[anchor] = False
        D[anchor] = -1
        P[anchor] = -1

    return stats


def enumerate_connected_subgraphs(neighbor_map, max_size=None):
    """Enumerate connected induced subgraphs up to an optional atom limit."""
    yield from iter_connected_subgraph_bitmasks(neighbor_map, max_size=max_size)


def subgraph_to_pyg_data(mining_graph, subgraph_mask):
    """Convert selected atoms into a masked full-graph PyTorch Geometric sample."""
    atom_count = mining_graph["atom_count"]
    node_mask = torch.zeros((atom_count, 1), dtype=torch.float32)

    for atom_idx in iter_set_bits(subgraph_mask):
        node_mask[atom_idx, 0] = 1.0

    graph_data = DATA.Data(
        x=mining_graph["full_x_tensor"],
        edge_index=mining_graph["edge_index_tensor"],
    )

    graph_data.node_mask = node_mask
    graph_data.mol_features = mining_graph["mol_features_tensor"]
    graph_data.__setitem__("c_size", torch.LongTensor([atom_count]))

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

    for candidate in sorted(
        sufficient_subgraphs,
        key=lambda item: (item["size"], item["difference"]),
    ):
        candidate_mask = candidate["mask"]

        if any((other["mask"] & candidate_mask) == other["mask"] for other in minimal_subgraphs):
            continue

        minimal_subgraphs = [
            other
            for other in minimal_subgraphs
            if not ((candidate_mask & other["mask"]) == candidate_mask)
        ]
        minimal_subgraphs.append(candidate)

    return minimal_subgraphs


def describe_subgraph_atoms(mining_graph, subgraph_mask):
    """Return atom symbols and simple structural flags for a subgraph."""
    mol = mining_graph["mol"]

    atom_descriptions = []
    for atom_idx in bitmask_to_atoms(subgraph_mask):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_descriptions.append({
            "atom_idx": atom_idx,
            "symbol": atom.GetSymbol(),
            "is_aromatic": atom.GetIsAromatic(),
            "is_in_ring": atom.IsInRing(),
        })

    return atom_descriptions


def subgraph_to_smiles(mining_graph, subgraph_mask):
    """Convert selected atom indices into an RDKit fragment SMILES string."""
    return Chem.MolFragmentToSmiles(
        mining_graph["mol"],
        atomsToUse=bitmask_to_atoms(subgraph_mask),
        canonical=True,
        isomericSmiles=True,
    )


def union_subgraph_combination(subgraph_combination):
    """Return the union of atom indices from a subgraph combination."""
    union_mask = 0

    for mask in subgraph_combination:
        union_mask |= mask

    return union_mask


def generate_pairwise_disjoint_combinations(
    subgraph_masks,
    max_combination_size=2,
    max_total_atoms=None,
):
    """Generate pairwise-disjoint subgraph combinations under the given size limits."""
    if max_combination_size not in (1, 2):
        raise NotImplementedError(
            "current streaming version is written for max_combination_size <= 2"
        )

    seen_masks = []

    for mask in subgraph_masks:
        if max_total_atoms is None or mask.bit_count() <= max_total_atoms:
            yield mask, (mask,)

        if max_combination_size >= 2:
            for previous_mask in seen_masks:
                if previous_mask & mask:
                    continue

                union_mask = previous_mask | mask
                if max_total_atoms is not None and union_mask.bit_count() > max_total_atoms:
                    continue

                yield union_mask, (previous_mask, mask)

        seen_masks.append(mask)


def deduplicate_sufficient_subgraphs_by_atoms(sufficient_subgraphs):
    """Remove duplicate explanations with the same final atom set."""
    best_by_mask = {}

    for item in sufficient_subgraphs:
        mask_key = item["mask"]

        if mask_key not in best_by_mask:
            best_by_mask[mask_key] = item
            continue

        existing = best_by_mask[mask_key]

        existing_components = existing.get("num_components", 1)
        current_components = item.get("num_components", 1)

        # prefer the simpler explanation first, then the closer prediction
        if current_components < existing_components:
            best_by_mask[mask_key] = item
        elif (
            current_components == existing_components
            and item["difference"] < existing["difference"]
        ):
            best_by_mask[mask_key] = item

    return list(best_by_mask.values())


def format_subgraph_components(subgraph_combination):
    """Format subgraph components as sorted atom-index lists."""
    return [bitmask_to_atoms(mask) for mask in subgraph_combination]
