import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem, RDConfig
from rdkit.Chem import MolFromSmiles, Descriptors, rdMolDescriptors, ChemicalFeatures
import networkx as nx
from utils import *
from kinase_features import build_all_target_feature_tables

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

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

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
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index, mol_features

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  

def seq_cat_unk(prot, max_len):
    x = np.zeros(max_len)
    for i, ch in enumerate(str(prot)[:max_len]):
        x[i] = seq_dict.get(ch, 0)
    return x

# one-hot encoding for protein features
def one_hot_value(value, vocab):
    return [1.0 if value == v else 0.0 for v in vocab]
# one numeric feature [V] per target_id
def build_target_feature_map(feature_df):
    feature_df = feature_df.fillna('')

    group_vocab = sorted(feature_df['kinase_group'].unique())
    family_vocab = sorted(feature_df['kinase_family'].unique())
    subfamily_vocab = sorted(feature_df['kinase_subfamily'].unique())
    dfg_vocab = sorted(feature_df['dfg_state'].unique())
    ac_vocab = sorted(feature_df['ac_helix_state'].unique())

    target_feat_map = {}

    for _, row in feature_df.iterrows():
        feat = []
        feat += [float(row['has_structure']) if str(row['has_structure']).strip() != '' else 0.0]
        feat += [float(row['pocket_hydrophobicity']) if str(row['pocket_hydrophobicity']).strip() != '' else 0.0]
        feat += [float(row['pocket_charge']) if str(row['pocket_charge']).strip() != '' else 0.0]
        # iter 2, added to protein_feat
        feat += [float(row['activation_loop_state']) if str(row['activation_loop_state']).strip() != '' else 0.0]
        feat += one_hot_value(row['kinase_group'], group_vocab)
        feat += one_hot_value(row['kinase_family'], family_vocab)
        feat += one_hot_value(row['kinase_subfamily'], subfamily_vocab)
        feat += one_hot_value(row['dfg_state'], dfg_vocab)
        feat += one_hot_value(row['ac_helix_state'], ac_vocab)

        target_feat_map[row['target_id']] = np.asarray(feat, dtype=np.float32)
    return target_feat_map

# build feature maps for klifs_85_residue_sequence, gatekeeper_residue, hinge_residues (protein feature iter 2)
def build_target_sequence_maps(feature_df):
    feature_df = feature_df.fillna('')

    pocket_map = {}
    gatekeeper_map = {}
    hinge_map = {}

    for _, row in feature_df.iterrows():
        target_id = row['target_id']
        pocket_map[target_id] = seq_cat_unk(row['klifs_85_residue_sequence'], 85)
        gatekeeper_map[target_id] = seq_cat_unk(row['gatekeeper_residue'], 1)
        hinge_map[target_id] = seq_cat_unk(row['hinge_residues'], 3)

    return pocket_map, gatekeeper_map, hinge_map

# from DeepDTA data
all_prots = []
datasets = ['kiba','davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')

    drugs = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)

    # to add additional features, we don't drop prot_id
    prot_ids = []
    prot_seqs = []
    for t in proteins.keys():
        prot_ids.append(t)
        prot_seqs.append(proteins[t])

    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train','test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_id,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prot_ids[cols[pair_ind]] ]
                ls += [ prot_seqs[cols[pair_ind]] ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prot_ids)))
    all_prots += list(set(prot_seqs))

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# trigger protein-feature table build
build_all_target_feature_tables()

compound_iso_smiles = []
for dt_name in ['kiba','davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis','kiba']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        # read protein features
        target_feature_df = pd.read_csv('data/' + dataset + '_target_features.csv')
        target_feature_map = build_target_feature_map(target_feature_df)
        pocket_map, gatekeeper_map, hinge_map = build_target_sequence_maps(target_feature_df)

        df = pd.read_csv('data/' + dataset + '_train.csv')
        # structure change: now stores target_id as well
        train_drugs,train_target_ids,train_prots,train_Y = list(df['compound_iso_smiles']),list(df['target_id']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in train_prots]
        XK = [pocket_map[tid] for tid in train_target_ids]
        XG = [gatekeeper_map[tid] for tid in train_target_ids]
        XH = [hinge_map[tid] for tid in train_target_ids]
        # map each target_id to the correspoding row in feature table
        XP = [target_feature_map[tid] for tid in train_target_ids]
        train_drugs,train_prots,train_feat,train_pocket,train_gatekeeper,train_hinge,train_Y = np.asarray(train_drugs),np.asarray(XT),np.asarray(XP, dtype=np.float32),np.asarray(XK),np.asarray(XG),np.asarray(XH),np.asarray(train_Y)

        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs,test_target_ids,test_prots,test_Y = list(df['compound_iso_smiles']),list(df['target_id']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in test_prots]
        # separate tensors for residues-related protein features
        XK = [pocket_map[tid] for tid in test_target_ids]
        XG = [gatekeeper_map[tid] for tid in test_target_ids]
        XH = [hinge_map[tid] for tid in test_target_ids]
        XP = [target_feature_map[tid] for tid in test_target_ids]
        test_drugs,test_prots,test_feat,test_pocket,test_gatekeeper,test_hinge,test_Y = np.asarray(test_drugs),np.asarray(XT),np.asarray(XP, dtype=np.float32),np.asarray(XK),np.asarray(XG),np.asarray(XH),np.asarray(test_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train',
                                    xd=train_drugs, xt=train_prots, xp=train_feat,
                                    xk=train_pocket, xg=train_gatekeeper, xh=train_hinge,
                                    y=train_Y,smile_graph=smile_graph)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test',
                                   xd=test_drugs, xt=test_prots, xp=test_feat,
                                   xk=test_pocket, xg=test_gatekeeper, xh=test_hinge,
                                   y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')