import os

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, ChemicalFeatures

"""
Ligand feature construction utilities.

Provides atom-level pharmacophore/chemical features and molecule-level
descriptor features for GraphDTA ligand preprocessing.
"""
fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# Builds a per-atom boolean pharmacophore matrix from RDKit chemical features.
# Columns: [Donor, Acceptor, Hydrophobe/LumpedHydrophobe, Aromatic, PosIonizable, NegIonizable]
def pharmacophore_atom_flags(mol):
    flags = np.zeros((mol.GetNumAtoms(), 6), dtype=bool)

    for feature in feature_factory.GetFeaturesForMol(mol):
        family = feature.GetFamily()

        for atom_id in feature.GetAtomIds():
            if family == 'Donor':
                flags[atom_id][0] = True
            elif family == 'Acceptor':
                flags[atom_id][1] = True
            elif family in ['Hydrophobe', 'LumpedHydrophobe']:
                flags[atom_id][2] = True
            elif family == 'Aromatic':
                flags[atom_id][3] = True
            elif family == 'PosIonizable':
                flags[atom_id][4] = True
            elif family == 'NegIonizable':
                flags[atom_id][5] = True

    return flags

# This function one-hot-encodes / numerically encodes the features of a single ligand atom.
#
# Original atom features:
# - atom type
# - atom degree: number of directly bonded neighboring atoms
# - total number of hydrogens bonded to the atom
# - implicit valence
# - aromaticity
#
# Added atom features:
# - formal charge
# - hybridization
# - ring membership: whether the atom is part of a ring
# - heteroatom flag: whether the atom is not carbon or hydrogen
# - halogen flag: whether the atom is F, Cl, Br, or I
# - pharmacophore flags: donor, acceptor, hydrophobe, aromatic, positive ionizable, negative ionizable
def atom_features(atom, pharm_flags):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 'Other']) +
                    one_of_k_encoding_unk(str(atom.GetHybridization()),['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Other']) +
                    [atom.GetIsAromatic(),
                     atom.IsInRing(),
                     atom.GetSymbol() not in ['C', 'H'],
                     atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']] +
                    list(pharm_flags))

# This function creates a molecule-level/global ligand feature vector.
# Unlike atom_features(atom), these descriptors describe the whole ligand molecule,
# not a single atom or bond.
#
# Added molecule-level features:
# - molecular weight
# - logP: lipophilicity / hydrophobicity estimate
# - TPSA: topological polar surface area
# - number of rotatable bonds
# - number of H-bond donors
# - number of H-bond acceptors
# - number of heteroatoms
# - number of rings
def molecule_features(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHeteroatoms(mol),
        rdMolDescriptors.CalcNumRings(mol),
    ])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
