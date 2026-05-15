import json
import os
import re
import pandas as pd
import urllib.parse
import urllib.request
import urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

TARGET_FEATURE_COLUMNS = [
    "target_id",
    "target_sequence",
    "kinase_group",
    "kinase_family",
    "kinase_subfamily",
    "klifs_kinase_id",
    "best_structure_klifs_id",
    "best_structure_pdb_id",
    "has_structure",
    "klifs_85_residue_sequence",
    "gatekeeper_residue",
    "hinge_residues",
    "pocket_residue_composition",
    "pocket_hydrophobicity",
    "pocket_charge",
    "dfg_state",
    "ac_helix_state",
    "activation_loop_state",
]

AA_HYDROPHOBIC = set(list("AVLIMFWYPCG"))
AA_POSITIVE = set(list("KRH"))
AA_NEGATIVE = set(list("DE"))
KLIFS_GATEKEEPER_INDEX = 45
KLIFS_HINGE_START = 46
KLIFS_HINGE_END = 48
KLIFS_API_BASE = "https://klifs.net/api"
KLIFS_API_CACHE = {}
KLIFS_STRUCTURE_CACHE = {}
KLIFS_CACHE_LOCK = Lock()

MANUAL_KLIFS_IDS = {
}

MANUAL_NAME_MAP = {
    "AMPK-ALPHA1": "PRKAA1",
    "AMPK-ALPHA2": "PRKAA2",
    "ARK5": "NUAK1",
    "ASK1": "MAP3K5",
    "ASK2": "MAP3K6",
    "BIKE": "BMP2K",
    "CDC2": "CDK1",
    "CDC2L1": "CDK11B",
    "CDC2L2": "CDK11A",
    "CDC2L5": "CDK13",
    "DCAMKL2": "DCLK2",
    "DCAMKL3": "DCLK3",
    "ERK1": "MAPK3",
    "ERK2": "MAPK1",
    "ERK8": "MAPK15",
    "FAK": "PTK2",
    "FAK2": "PTK2B",
    "FGFR1OP2-FGFR1": "FGFR1",
    "IKK-EPSILON": "IKBKE",
    "JNK1": "MAPK8",
    "JNK2": "MAPK9",
    "JNK3": "MAPK10",
    "MEK1": "MAP2K1",
    "MEK2": "MAP2K2",
    "MEK3": "MAP2K3",
    "MEK4": "MAP2K4",
    "MEK5": "MAP2K5",
    "MKK4": "MAP2K4",
    "MKK6": "MAP2K6",
    "MKK7": "MAP2K7",
    "MLCK": "MYLK",
    "MST1": "STK4",
    "MST2": "STK3",
    "NEK11": "NEK11",
    "PAK7": "PAK5",
    "P38-ALPHA": "MAPK14",
    "P38-BETA": "MAPK11",
    "P38-DELTA": "MAPK13",
    "P38-GAMMA": "MAPK12",
    "PCTK1": "CDK16",
    "PCTK2": "CDK17",
    "PCTK3": "CDK18",
    "PDGFR-ALPHA": "PDGFRA",
    "PDGFR-BETA": "PDGFRB",
    "PFTAIRE2": "CDK17",
    "PFTK1": "CDK14",
    "PIP5K2B": "PIP4K2B",
    "PIP5K2C": "PIP4K2C",
    "PKAC-ALPHA": "PRKACA",
    "PKAC-BETA": "PRKACB",
    "PKG1": "PRKG1",
    "PKN1": "PKN1",
    "PKN2": "PKN2",
    "PKN3": "PKN3",
    "PRKR": "EIF2AK2",
    "RAF1": "RAF1",
    "RIPK2": "RIPK2",
    "RIPK5": "DSTYK",
    "RSK1": "RPS6KA1",
    "RSK2": "RPS6KA3",
    "RSK3": "RPS6KA2",
    "RSK4": "RPS6KA6",
    "S6K1": "RPS6KB1",
    "S6K2": "RPS6KB2",
    "SNARK": "NUAK2",
    "SRC(F317L)": "SRC",
    "SYK": "SYK",
    "TNIK": "TNIK",
    "TRKA": "NTRK1",
    "TRKB": "NTRK2",
    "TRKC": "NTRK3",
    "YSK4": "MAP3K19",
}

def empty_feature_row(target_id, target_sequence) :
    return {
        "target_id": target_id,
        "target_sequence": target_sequence,
        "kinase_group": "",
        "kinase_family": "",
        "kinase_subfamily": "",
        "klifs_kinase_id": "",
        "best_structure_klifs_id": "",
        "best_structure_pdb_id": "",
        "has_structure": 0,
        "klifs_85_residue_sequence": "",
        "gatekeeper_residue": "",
        "hinge_residues": "",
        "pocket_residue_composition": "",
        "pocket_hydrophobicity": "",
        "pocket_charge": "",
        "dfg_state": "",
        "ac_helix_state": "",
        "activation_loop_state": "",
    }

def choose_best_structure_from_api(structures):
    if not structures:
        return None

    def ranking_key(structure):
        try:
            quality_score = float(structure.get("quality_score") or float("-inf"))
        except Exception:
            quality_score = float("-inf")
        try:
            resolution = float(structure.get("resolution") or float("inf"))
        except Exception:
            resolution = float("inf")
        return (-quality_score, resolution)

    return sorted(structures, key=ranking_key)[0]

def klifs_api_get(path, **params):
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{KLIFS_API_BASE}{path}?{query}" if query else f"{KLIFS_API_BASE}{path}"
    with KLIFS_CACHE_LOCK:
        if url in KLIFS_API_CACHE:
            return KLIFS_API_CACHE[url]

    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    with KLIFS_CACHE_LOCK:
        if url not in KLIFS_API_CACHE:
            KLIFS_API_CACHE[url] = data
        return KLIFS_API_CACHE[url]

def normalize_klifs_sequence(klifs_seq):
    if pd.isna(klifs_seq):
        return ""
    klifs_seq = str(klifs_seq).strip().replace("-", "_")
    return klifs_seq if len(klifs_seq) == 85 else klifs_seq

def compute_pocket_summary(klifs_seq):
    aa_only = [aa for aa in klifs_seq if aa.isalpha() and aa != "_"]
    if not aa_only:
        return "", 0.0, 0.0

    hydro = sum(aa in AA_HYDROPHOBIC for aa in aa_only)/len(aa_only)
    charge = (
        sum(aa in AA_POSITIVE for aa in aa_only) -
        sum(aa in AA_NEGATIVE for aa in aa_only)
    )/len(aa_only)

    counts = {}
    for aa in aa_only:
        counts[aa] = counts.get(aa, 0) + 1
    composition = ";".join(f"{aa}:{counts[aa]}" for aa in sorted(counts))

    return composition, hydro, charge

def extract_gatekeeper_and_hinge(klifs_seq):
    klifs_seq = normalize_klifs_sequence(klifs_seq)
    if len(klifs_seq) < KLIFS_HINGE_END:
        return "", ""

    gatekeeper = klifs_seq[KLIFS_GATEKEEPER_INDEX - 1]
    hinge_residues = klifs_seq[KLIFS_HINGE_START - 1:KLIFS_HINGE_END]
    return gatekeeper, hinge_residues

def populate_row_from_structure_fields(row, structure_fields):
    klifs_seq = normalize_klifs_sequence(structure_fields.get("pocket", ""))
    composition, hydro, charge = compute_pocket_summary(klifs_seq)
    gatekeeper, hinge = extract_gatekeeper_and_hinge(klifs_seq)

    row["best_structure_klifs_id"] = str(structure_fields.get("structure_ID", row.get("best_structure_klifs_id", "")) or "")
    row["best_structure_pdb_id"] = str(structure_fields.get("pdb", row.get("best_structure_pdb_id", "")) or "")
    row["has_structure"] = "1"
    row["klifs_85_residue_sequence"] = klifs_seq
    row["gatekeeper_residue"] = gatekeeper
    row["hinge_residues"] = hinge
    row["pocket_residue_composition"] = composition
    row["pocket_hydrophobicity"] = str(hydro)
    row["pocket_charge"] = str(charge)
    row["dfg_state"] = str(structure_fields.get("DFG", row.get("dfg_state", "")) or "")
    row["ac_helix_state"] = str(structure_fields.get("aC_helix", row.get("ac_helix_state", "")) or "")
    row["activation_loop_state"] = str(structure_fields.get("Grich_rotation", row.get("activation_loop_state", "")) or "")
    return row

# separate per-target table 
# each row: taget_id, taget_sequence, other_features
def load_targets_from_proteins_file(dataset):
    proteins_path = os.path.join("data", dataset, "proteins.txt")
    proteins = json.load(open(proteins_path), object_pairs_hook=OrderedDict)
    return list(proteins.items())  

def normalize_target_id(target_id):
    cleaned = target_id.strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    if cleaned.endswith("p") and "(" in cleaned:
        cleaned = cleaned[:-1]
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = cleaned.replace("/", "-")
    cleaned_upper = cleaned.upper()
    return MANUAL_NAME_MAP.get(cleaned_upper, cleaned_upper)

def resolve_kinase_info(target_id):
    if target_id in MANUAL_KLIFS_IDS:
        try:
            kinase_rows = klifs_api_get("/kinase_information", kinase_ID=MANUAL_KLIFS_IDS[target_id], species="HUMAN")
        except urllib.error.HTTPError:
            return None
        return kinase_rows[0] if kinase_rows else None

    query_name = normalize_target_id(target_id)
    try:
        kinase_rows = klifs_api_get("/kinase_ID", kinase_name=query_name, species="HUMAN")
    except urllib.error.HTTPError:
        return None
    return kinase_rows[0] if kinase_rows else None

def row_is_resolved(row):
    value = row.get("klifs_kinase_id", "")
    return pd.notna(value) and str(value).strip() != ""

def fill_from_sequence_matches(df, donor_dfs):
    donor_rows = []
    for donor_df in donor_dfs:
        if donor_df is None or donor_df.empty:
            continue
        donor_rows.extend(donor_df.to_dict(orient="records"))

    seq_to_donor = {}
    for donor_row in donor_rows:
        if not row_is_resolved(donor_row):
            continue
        target_sequence = donor_row.get("target_sequence", "")
        if target_sequence and target_sequence not in seq_to_donor:
            seq_to_donor[target_sequence] = donor_row

    unresolved_mask = df["klifs_kinase_id"].isna() | (df["klifs_kinase_id"].astype(str).str.strip() == "")
    unresolved_indices = df[unresolved_mask].index.tolist()

    for idx in unresolved_indices:
        target_sequence = df.at[idx, "target_sequence"]
        donor_row = seq_to_donor.get(target_sequence)
        if donor_row is None:
            continue
        for col in TARGET_FEATURE_COLUMNS:
            if col in {"target_id", "target_sequence"}:
                continue
            df.at[idx, col] = donor_row.get(col, df.at[idx, col])

    return df

def load_existing_feature_tables(current_dataset):
    donor_dfs = []
    for dataset in ["davis", "kiba"]:
        if dataset == current_dataset:
            continue
        path = os.path.join("data", f"{dataset}_target_features.csv")
        if os.path.exists(path):
            donor_dfs.append(pd.read_csv(path, dtype=str).fillna(""))
    return donor_dfs

def fetch_klifs_features(target_id, target_sequence):
    row = empty_feature_row(target_id, target_sequence) # safe default-fallback

    kinase_info = resolve_kinase_info(target_id)
    if kinase_info is None:
        return row

    kinase_klifs_id = str(kinase_info.get("kinase_ID", "") or "")
    row["klifs_kinase_id"] = kinase_klifs_id
    row["kinase_family"] = str(kinase_info.get("family", "") or "")
    row["kinase_group"] = str(kinase_info.get("group", "") or "")
    row["kinase_subfamily"] = str(kinase_info.get("kinase_class", "") or "")

    if not kinase_klifs_id:
        return row

    with KLIFS_CACHE_LOCK:
        cached_structures = KLIFS_STRUCTURE_CACHE.get(kinase_klifs_id)

    if cached_structures is None:
        try:
            cached_structures = klifs_api_get("/structures_list", kinase_ID=kinase_klifs_id)
        except urllib.error.HTTPError:
            cached_structures = []

        with KLIFS_CACHE_LOCK:
            if kinase_klifs_id not in KLIFS_STRUCTURE_CACHE:
                KLIFS_STRUCTURE_CACHE[kinase_klifs_id] = cached_structures
            cached_structures = KLIFS_STRUCTURE_CACHE[kinase_klifs_id]

    best_structure = choose_best_structure_from_api(cached_structures)
    if best_structure is None:
        return row

    row = populate_row_from_structure_fields(row, best_structure)
    return row

def fetch_target_row(dataset, target_id, target_sequence):
    try:
        return fetch_klifs_features(target_id, target_sequence)
    except Exception as e:
        print(f"[warn] {dataset} {target_id}: {e}")
        return empty_feature_row(target_id, target_sequence)

# safely parallelize (preserve row order)
def build_row_parallel(dataset, targets, max_workers=0):
    rows_by_index = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fetch_target_row, dataset, target_id, target_sequence): i
            for i, (target_id, target_sequence) in enumerate(targets)
        }
        
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            rows_by_index[i] = future.result()
    return [rows_by_index[i] for i in range(len(targets))]
      
# tries to fetch features from klifs via opencadd by target_id
# if target_id could not be resolved, fill the row with default values
def build_target_feature_table(dataset, force=False):
    output_path = os.path.join("data", f"{dataset}_target_features.csv")
    if os.path.exists(output_path) and not force:
        print(f"skip existing {output_path}")
        return

    rows = []
    
    targets = load_targets_from_proteins_file(dataset)
    rows = build_row_parallel(dataset, targets, max_workers=6)
    
    df = pd.DataFrame(rows, columns=TARGET_FEATURE_COLUMNS).fillna("")
    donor_dfs = load_existing_feature_tables(dataset)
    donor_dfs.append(df[df["klifs_kinase_id"].astype(str).str.strip() != ""].copy())
    df = fill_from_sequence_matches(df, donor_dfs)
    df.to_csv(output_path, index=False)
    print(f"wrote {output_path} with {len(df)} targets")

def build_all_target_feature_tables(force=False):
    for dataset in ["kiba", "davis"]:
        build_target_feature_table(dataset, force=force)
    
    
def main():
    build_all_target_feature_tables()
        
if __name__ == "__main__":
    main()            
