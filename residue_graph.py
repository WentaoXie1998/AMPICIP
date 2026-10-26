'''
对于每个残基（一个氨基酸字母 aa）+ 它的二级结构 ss8 + SASA 值，我们拼成：
残基类型 one-hot（21 维：20 个氨基酸 + 'X'）
疏水性 1：hydrophobicity@pH7（1 维）
电荷（-1/0/+1 简化）（1 维）
极性（极性=1，非极性=0）（1 维）
二级结构 SS8 one-hot（8 维）
SASA（溶剂可及表面积，1 维）
芳香性 aromatic（1 维）
“体积”：这里用残基相对分子质量 res_weight_table 近似（1 维）
pI（等电点 res_pl_table，1 维）
Hydrophobicity-2：用你给的 res_hydrophobic_ph2_table（1 维）
Kidera 10 维（kidera_feature_table）

永远都是 47 维 “有 / 没有 DSSP”只影响的是：SS8（二级结构 8 维）+ SASA（1 维）这 9 维里是“真实值”还是“全 0 占位”，维度本身不会变。

残基边特征（每条边 2 维） 在 build_residue_graph 里构造：
CA_dist：两个残基 CA 原子的欧氏距离，连续实数
is_backbone：是否主链相邻（|i-j| == 1 时为 1，否则 0）

这里面的build_pocket_residue_subgraphs_precomputed() 生成 pocket，并最终拼成“诱导子图”
'''

from __future__ import annotations

import os
import math
import hashlib
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import torch


AA_LIST = [
    "A","C","D","E","F","G","H","I","K","L",
    "M","N","P","Q","R","S","T","V","W","Y","X"
]  # X: unknown


KD_HYDRO = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
    "X": 0.0,
}

CHARGE_TABLE = {
    "D": -1.0, "E": -1.0,
    "K":  1.0, "R":  1.0, "H":  0.5,
}

POLAR_SET = set(["C","N","Q","S","T","D","E","K","R","H","Y"])

AROMATIC_SET = set(["F","W","Y"])

RES_VOLUME = {
    "A": 88.6,  "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1,  "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0,  "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
    "X": 0.0,
}

res_pl_table = {
    "A": 6.00, "C": 5.07, "D": 2.77, "E": 3.22, "F": 5.48,
    "G": 5.97, "H": 7.59, "I": 6.02, "K": 9.74, "L": 5.98,
    "M": 5.74, "N": 5.41, "P": 6.30, "Q": 5.65, "R": 10.76,
    "S": 5.68, "T": 5.60, "V": 5.96, "W": 5.89, "Y": 5.96,
    "X": 0.00,
}

res_hydrophobic_ph2_table = {
    "A": 47,  "C": 52,  "D": -18, "E": 8,   "F": 92,
    "G": 0,   "H": -42, "I": 100, "K": -37, "L": 100,
    "M": 74,  "N": -41, "P": -46, "Q": -18, "R": -26,
    "S": -7,  "T": 13,  "V": 79,  "W": 84,  "Y": 49,
    "X": 0,
}

kidera_feature_table = {
    "A": [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.20, -0.08,  0.21, -0.48],
    "C": [ 0.12, -0.89,  0.45, -1.05, -0.71,  2.41,  1.52, -0.69,  1.13,  1.10],
    "E": [-1.45,  0.19, -1.61,  1.17, -1.31,  0.40,  0.04,  0.38, -0.35, -0.12],
    "D": [ 0.58, -0.22, -1.58,  0.81, -0.92,  0.15, -1.52,  0.47,  0.76,  0.70],
    "G": [ 1.46, -1.96, -0.23, -0.16,  0.10, -0.11,  1.32,  2.36, -1.66,  0.46],
    "F": [-0.21,  0.98, -0.36, -1.43,  0.22, -0.81,  0.67,  1.10,  1.71, -0.44],
    "I": [-0.73, -0.16,  1.79, -0.77, -0.54,  0.03, -0.83,  0.51,  0.66, -1.78],
    "H": [-0.41,  0.52, -0.28,  0.28,  1.61,  1.01, -1.85,  0.47,  1.13,  1.63],
    "K": [-0.34,  0.82, -0.23,  1.70,  1.54, -1.62,  1.15, -0.08, -0.48,  0.60],
    "M": [-1.40,  0.18, -0.42, -0.73,  2.00,  1.52,  0.26,  0.11, -1.27,  0.27],
    "L": [-1.04,  0.00, -0.24, -1.10, -0.55, -2.05,  0.96, -0.76,  0.45,  0.93],
    "N": [ 1.14, -0.07, -0.12,  0.81,  0.18,  0.37, -0.09,  1.23,  1.10, -1.73],
    "Q": [-0.47,  0.24,  0.07,  1.10,  1.10,  0.59,  0.84, -0.71, -0.03, -2.33],
    "P": [ 2.06, -0.33, -1.15, -0.75,  0.88, -0.45,  0.30, -2.30,  0.74, -0.28],
    "S": [ 0.81, -1.08,  0.16,  0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
    "R": [ 0.22,  1.27,  1.37,  1.87, -1.70,  0.46,  0.92, -0.39,  0.23,  0.93],
    "T": [ 0.26, -0.70,  1.21,  0.63, -0.10,  0.21,  0.24, -1.15, -0.56,  0.19],
    "W": [ 0.30,  2.10, -0.72, -1.57, -1.16,  0.57, -0.48, -0.40, -2.30, -0.60],
    "V": [-0.74, -0.71,  2.04, -0.40,  0.50, -0.81, -1.07,  0.06, -0.46,  0.65],
    "Y": [ 1.38,  1.48,  0.80, -0.56, -0.00, -0.68, -0.31,  1.03, -0.05,  0.53],
    "X": [0.0] * 10,
}


def aa_one_hot(res: str) -> np.ndarray:
    if res not in AA_LIST:
        res = "X"
    vec = np.zeros(len(AA_LIST), dtype=np.float32)
    vec[AA_LIST.index(res)] = 1.0
    return vec

SS8_SYMBOLS = ["H", "B", "E", "G", "I", "T", "S", "C"]  # C: coil/other

def ss8_one_hot(ss_char: str) -> np.ndarray:
    if ss_char == " ":
        ss_char = "C"
    if ss_char not in SS8_SYMBOLS:
        ss_char = "C"
    vec = np.zeros(len(SS8_SYMBOLS), dtype=np.float32)
    vec[SS8_SYMBOLS.index(ss_char)] = 1.0
    return vec

def encode_residue_feature(
    res: str,
    ss8_char: Optional[str] = None,
    sasa: Optional[float] = None,
) -> np.ndarray:
    """
    47-d:
      21 type + 1 hydro + 1 charge + 1 polar +
      8 SS8 + 1 SASA + 1 aromatic + 1 volume +
      1 pI + 1 Hydro2 + 10 Kidera
    """
    res = res if res in AA_LIST else "X"

    f_type   = aa_one_hot(res)  # 21
    f_hydro  = np.array([KD_HYDRO.get(res, 0.0)], dtype=np.float32)         # 1
    f_charge = np.array([CHARGE_TABLE.get(res, 0.0)], dtype=np.float32)     # 1
    f_polar  = np.array([1.0 if res in POLAR_SET else 0.0], dtype=np.float32)  # 1

    if ss8_char is not None:
        f_ss8 = ss8_one_hot(ss8_char)  # 8
    else:
        f_ss8 = np.zeros(len(SS8_SYMBOLS), dtype=np.float32)

    if sasa is not None:
        try:
            sasa_val = float(sasa)
        except Exception:
            sasa_val = 0.0
        f_sasa = np.array([sasa_val], dtype=np.float32)  # 1
    else:
        f_sasa = np.array([0.0], dtype=np.float32)       # 1

    f_arom = np.array([1.0 if res in AROMATIC_SET else 0.0], dtype=np.float32)  # 1
    f_vol  = np.array([RES_VOLUME.get(res, 0.0)], dtype=np.float32)             # 1
    f_pl   = np.array([res_pl_table.get(res, 0.0)], dtype=np.float32)           # 1
    f_h2   = np.array([res_hydrophobic_ph2_table.get(res, 0.0)], dtype=np.float32)  # 1
    f_kid  = np.array(kidera_feature_table.get(res, [0.0]*10), dtype=np.float32)    # 10

    feat = np.concatenate(
        [f_type, f_hydro, f_charge, f_polar, f_ss8, f_sasa, f_arom, f_vol, f_pl, f_h2, f_kid],
        axis=0
    )
    assert feat.shape[0] == 47
    return feat


def build_residue_features_from_sequence(seq: str) -> np.ndarray:
    """
    No structure: SS8/SASA (9 dims) filled with 0, total stays 47.
    """
    feats = [encode_residue_feature(aa, None, None) for aa in seq]
    return np.stack(feats, axis=0)  # (L,47)



try:
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    HAS_DSSP = True
except ImportError:
    HAS_DSSP = False
    PDBParser = None
    DSSP = None


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cryst1_placeholder_line() -> str:
    """
    Strict fixed-width CRYST1 line for mkdssp tolerance (NOT a real unit cell):
      a=b=c=1.0; alpha=beta=gamma=90.0; space group 'P 1'; Z=1
    """
    a = b = c = 1.0
    alpha = beta = gamma = 90.0
    sg = "P 1"
    z = 1
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {sg:<11}{z:4d}"


def _norm_pdb_line_80(s: str) -> str:
    s = s.rstrip("\n")
    if len(s) < 80:
        s = s.ljust(80)
    else:
        s = s[:80]
    return s + "\n"


def _extract_atom_block_first_model(lines: List[str]) -> List[str]:
    has_model = any(l.startswith("MODEL") for l in lines)
    kept: List[str] = []

    if has_model:
        in_first = False
        seen_first = False
        for l in lines:
            rec = l[:6].strip()
            if rec == "MODEL":
                num_txt = l[10:14].strip()
                if not seen_first and (num_txt == "" or num_txt == "1"):
                    in_first = True
                    seen_first = True
                else:
                    in_first = False
                continue
            if rec == "ENDMDL":
                if in_first:
                    break
                continue

            if not in_first:
                continue
            if l.startswith(("ATOM", "TER")):
                kept.append(l)
    else:
        for l in lines:
            if l.startswith(("ATOM", "TER")):
                kept.append(l)

    return kept


def _make_sanitized_pdb_for_dssp(
    pdb_path: str,
    cache_dir: str = "/tmp/dssp_pdb_sanitized_cache",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)

    try:
        h = _sha1_file(pdb_path)
    except Exception:
        h = hashlib.sha1(pdb_path.encode("utf-8", errors="ignore")).hexdigest()

    base = Path(pdb_path).name
    out_path = os.path.join(cache_dir, f"{base}.{h}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    with open(pdb_path, "r", errors="replace") as f:
        lines = f.readlines()

    atom_lines = _extract_atom_block_first_model(lines)
    if not any(l.startswith("ATOM") for l in atom_lines):
        # no coordinates => return original (will fail later -> fallback)
        return pdb_path

    header = [
        "HEADER    SANITIZED FOR DSSP",
        f"TITLE     {base}",
        _cryst1_placeholder_line(),
    ]

    out_lines: List[str] = []
    for l in header:
        out_lines.append(_norm_pdb_line_80(l))

    for l in atom_lines:
        out_lines.append(_norm_pdb_line_80(l))

    # Ensure TER and END at the end
    if not any(x.startswith("TER") for x in out_lines[-8:]):
        out_lines.append(_norm_pdb_line_80("TER"))
    out_lines.append(_norm_pdb_line_80("END"))

    # Atomic write to avoid multi-worker corruption
    tmp_path = out_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.writelines(out_lines)
    os.replace(tmp_path, out_path)

    return out_path


def _parse_structure(pdb_path: str):
    if PDBParser is None:
        raise ImportError("PDBParser unavailable, please install biopython.")
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    return parser.get_structure("prot", pdb_path)


def _run_dssp(model, pdb_path: str):
    last_e = None
    for exe in ("mkdssp", "dssp", None):
        try:
            if exe is None:
                return DSSP(model, pdb_path)
            return DSSP(model, pdb_path, dssp=exe)
        except TypeError as e:
            last_e = e
            continue
        except Exception as e:
            last_e = e
            continue
    if last_e is not None:
        raise last_e
    raise RuntimeError("DSSP failed (unknown).")


def build_residue_features_from_pdb(
    seq: str,
    pdb_path: str,
    chain_id: str = "A",
    sanitize_cache_dir: str = "/tmp/dssp_pdb_sanitized_cache",
) -> np.ndarray:
    if not HAS_DSSP:
        print("[WARN] Biopython/DSSP not available, fallback to sequence features.")
        return build_residue_features_from_sequence(seq)

    fixed_pdb = _make_sanitized_pdb_for_dssp(pdb_path, cache_dir=sanitize_cache_dir)

    # parse structure from the SAME fixed_pdb so DSSP keys match
    try:
        structure = _parse_structure(fixed_pdb)
        model = structure[0]
    except Exception as e:
        print(f"[WARN] PDB parse failed on {pdb_path} (fixed={fixed_pdb}): {e}")
        print("[WARN] fallback to sequence features (SS8/SASA zeros).")
        return build_residue_features_from_sequence(seq)

    # chain fallback
    all_chains = [c.id for c in model.get_chains()]
    if len(all_chains) == 0:
        print(f"[WARN] No chains found in {pdb_path} after sanitize, fallback.")
        return build_residue_features_from_sequence(seq)
    if chain_id not in all_chains:
        print(f"[WARN] chain '{chain_id}' not in {pdb_path}, available={all_chains}, use '{all_chains[0]}'.")
        chain_id = all_chains[0]
    chain = model[chain_id]

    try:
        dssp = _run_dssp(model, fixed_pdb)
    except Exception as e:
        print(f"[WARN] DSSP failed on {pdb_path} (fixed={fixed_pdb}): {e}")
        print("[WARN] fallback to sequence features (SS8/SASA zeros).")
        return build_residue_features_from_sequence(seq)

    residues = [res for res in chain.get_residues() if res.id[0] == " "]
    L = min(len(seq), len(residues))
    if L != len(seq) or L != len(residues):
        print(f"[WARN] seq len {len(seq)} vs pdb residues {len(residues)} mismatch, use first {L}.")

    feats: List[np.ndarray] = []
    for i in range(L):
        aa = seq[i]
        res = residues[i]
        key = (chain_id, res.id)
        if key not in dssp:
            feats.append(encode_residue_feature(aa, None, None))
            continue
        d = dssp[key]
        ss_char = d[2]
        asa = d[3]
        feats.append(encode_residue_feature(aa, ss_char, asa))

    return np.stack(feats, axis=0)  # (L,47)


# ============================================================
# ========= 5. CA coordinates from PDB (use sanitized) =========
# ============================================================

def load_ca_coords_from_pdb(
    pdb_path: str,
    chain_id: str = "A",
    sanitize_cache_dir: str = "/tmp/dssp_pdb_sanitized_cache",
) -> Tuple[np.ndarray, List[Any]]:
    if PDBParser is None:
        raise ImportError("PDBParser unavailable, please install biopython.")

    fixed_pdb = _make_sanitized_pdb_for_dssp(pdb_path, cache_dir=sanitize_cache_dir)

    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("prot", fixed_pdb)
    model = structure[0]

    all_chains = [c.id for c in model.get_chains()]
    if len(all_chains) == 0:
        raise ValueError(f"No chains found in PDB: {pdb_path}")
    if chain_id not in all_chains:
        print(f"[WARN] chain '{chain_id}' not in {pdb_path}, available={all_chains}, use '{all_chains[0]}'.")
        chain_id = all_chains[0]
    chain = model[chain_id]

    coords: List[np.ndarray] = []
    residues: List[Any] = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        coords.append(res["CA"].coord)
        residues.append(res)

    coords_arr = np.array(coords, dtype=np.float32)
    return coords_arr, residues


def build_residue_graph(
    seq: str,
    pdb_path: str,
    chain_id: str = "A",
    dist_cutoff: float = 8.0,
    fully_connected: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build global residue contact graph using CA distances.

    Nodes: residues aligned by order (use first L=min(len(seq), #CA residues))
    Edges:
      - fully_connected=False: keep edges with d(i,j) < dist_cutoff
      - fully_connected=True:  keep all i!=j

    Edge features (2-d):
      [CA_dist, is_backbone]
        - CA_dist: Euclidean distance between CA atoms
        - is_backbone: 1 if |i-j|==1 else 0
    """
    coords, _residues = load_ca_coords_from_pdb(pdb_path, chain_id=chain_id)
    L_coords = coords.shape[0]
    L_seq = len(seq)
    L = min(L_coords, L_seq)

    if L_coords != L_seq:
        print(f"[WARN] seq len {L_seq} vs PDB CA count {L_coords} mismatch, use first {L}.")

    coords = coords[:L]
    seq = seq[:L]

    # distance matrix (L,L)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1).astype(np.float32)

    edges: List[List[int]] = []
    edge_feats: List[List[float]] = []

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            d = float(dist_matrix[i, j])

            if not fully_connected and d >= dist_cutoff:
                continue

            is_backbone = 1.0 if abs(i - j) == 1 else 0.0
            edges.append([i, j])
            edge_feats.append([d, is_backbone])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.LongTensor(edges).t().contiguous()   # (2,E)
        edge_attr = torch.FloatTensor(edge_feats)               # (E,2)

    return edge_index, edge_attr, dist_matrix


def _clip_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _topk_indices(p: np.ndarray, K: int) -> np.ndarray:
    K = int(_clip_int(K, 1, p.shape[0]))
    idx = np.argsort(p)[-K:][::-1]
    return idx.astype(np.int64)


def _entropy_concentration(p: np.ndarray) -> float:
    p = np.clip(p, 0.0, None).astype(np.float64)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    q = p / s
    eps = 1e-12
    H = -float(np.sum(q * np.log(q + eps)))
    H_max = math.log(len(q) + eps)
    H_norm = H / (H_max + eps)
    c = 1.0 - H_norm
    return float(np.clip(c, 0.0, 1.0))


def _rho_from_c(c: float, rho_min: float, rho_max: float) -> float:
    c = float(np.clip(c, 0.0, 1.0))
    rho = float(rho_min + (rho_max - rho_min) * c)
    return float(np.clip(rho, min(rho_min, rho_max), max(rho_min, rho_max)))


def _k_mass_coverage(p: np.ndarray, rho: float) -> int:
    p = np.clip(p, 0.0, None).astype(np.float64)
    total = float(p.sum())
    if total <= 0:
        return 1
    target = float(rho) * total
    p_sorted = np.sort(p)[::-1]
    cum = 0.0
    for i, val in enumerate(p_sorted, start=1):
        cum += float(val)
        if cum >= target:
            return i
    return len(p_sorted)


def _k_knee_lower_bound(p: np.ndarray) -> int:
    p = np.clip(p, 0.0, None).astype(np.float64)
    p_sorted = np.sort(p)[::-1]
    L = len(p_sorted)
    if L <= 2:
        return 1

    y = p_sorted / (p_sorted[0] + 1e-12)
    x = np.linspace(0.0, 1.0, L)

    y0, y1 = float(y[0]), float(y[-1])
    y_line = y0 + (y1 - y0) * x

    diff = y_line - y
    knee = int(np.argmax(diff)) + 1
    return int(_clip_int(knee, 1, L))


def _build_undirected_adj_from_edge_index(edge_index: torch.Tensor, L: int) -> List[List[int]]:
    adj: List[Set[int]] = [set() for _ in range(L)]
    if edge_index.numel() == 0:
        return [sorted(list(s)) for s in adj]

    ei = edge_index.detach().cpu().numpy()
    src = ei[0]
    dst = ei[1]
    for u, v in zip(src, dst):
        u = int(u); v = int(v)
        if 0 <= u < L and 0 <= v < L and u != v:
            adj[u].add(v)
            adj[v].add(u)

    return [sorted(list(s)) for s in adj]


def _bfs_hops(adj: List[List[int]], seeds: Set[int], max_hop: int) -> Set[int]:
    visited = set(seeds)
    frontier = set(seeds)

    for _ in range(int(max_hop)):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return visited


def _find_shortest_path_to_core(
    contact_adj: List[List[int]],
    start: int,
    core_set: Set[int],
    blocked: Optional[Set[int]] = None,
    max_depth: Optional[int] = None,
) -> Optional[List[int]]:
    if start in core_set:
        return [start]

    blocked = blocked or set()
    q = deque([start])
    parent = {start: -1}
    depth = {start: 0}

    while q:
        u = q.popleft()
        du = depth[u]
        if max_depth is not None and du >= int(max_depth):
            continue

        for v in contact_adj[u]:
            if v in parent:
                continue
            if v in blocked and v not in core_set:
                continue

            parent[v] = u
            depth[v] = du + 1

            if v in core_set:
                path = [v]
                cur = v
                while parent[cur] != -1:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path

            q.append(v)

    return None


def _enforce_connected_by_bridging(
    core_set: Set[int],
    added: List[int],
    contact_adj: List[List[int]],
    blocked: Optional[Set[int]] = None,
    max_depth: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int]]:
    blocked = blocked or set()
    kept_added: List[int] = []
    bridge_nodes: Set[int] = set()
    dropped_added: List[int] = []

    for v in added:
        v = int(v)
        if v in core_set:
            kept_added.append(v)
            continue

        path = _find_shortest_path_to_core(
            contact_adj=contact_adj,
            start=v,
            core_set=core_set,
            blocked=blocked,
            max_depth=max_depth,
        )
        if path is None:
            dropped_added.append(v)
            continue

        if len(path) >= 3:
            for mid in path[1:-1]:
                if mid not in core_set:
                    bridge_nodes.add(int(mid))

        kept_added.append(v)

    return kept_added, sorted(list(bridge_nodes)), dropped_added


def _connected_components_on_candidates_contact(
    cand_nodes: Set[int],
    contact_adj: List[List[int]],
) -> List[List[int]]:
    visited: Set[int] = set()
    comps: List[List[int]] = []

    for start in cand_nodes:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in contact_adj[u]:
                if v in cand_nodes and v not in visited:
                    visited.add(v)
                    stack.append(v)

        comps.append(sorted(comp))

    return comps


def _select_components_by_eta(
    comps: List[List[int]],
    p: np.ndarray,
    eta: float,
    topk_nodes: Set[int],
) -> List[List[int]]:
    eta = float(np.clip(eta, 0.0, 1.0))
    denom = float(np.sum([p[i] for i in topk_nodes])) if len(topk_nodes) > 0 else float(p.sum())
    if denom <= 0:
        comps_sorted = sorted(comps, key=lambda c: len(c), reverse=True)
        return comps_sorted[:1] if comps_sorted else []

    comp_masses = []
    for C in comps:
        mass = float(np.sum([p[i] for i in C]))
        comp_masses.append((mass, C))

    comp_masses.sort(key=lambda x: x[0], reverse=True)

    kept: List[List[int]] = []
    cum = 0.0
    target = eta * denom

    for mass, C in comp_masses:
        kept.append(C)
        cum += mass
        if cum >= target:
            break

    if not kept and comp_masses:
        kept = [comp_masses[0][1]]

    return kept


def _sigma_for_component(
    C: List[int],
    nbrs: List[int],
    dist_matrix: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    sigma_default: float,
) -> float:
    if len(C) == 0 or len(nbrs) == 0:
        return float(sigma_default)

    mins = []
    for v in nbrs:
        dv = [float(dist_matrix[u, v]) for u in C]
        mins.append(min(dv))

    med = float(np.median(np.array(mins, dtype=np.float32)))
    sigma = float(np.clip(med, sigma_min, sigma_max))
    return sigma


def _score_neighbors(
    C: List[int],
    nbrs: List[int],
    p: np.ndarray,
    dist_matrix: np.ndarray,
    sigma: float,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    if len(C) == 0:
        return {v: float(p[v]) for v in nbrs}

    sigma = float(max(sigma, 1e-6))
    for v in nbrs:
        best = 0.0
        for u in C:
            d = float(dist_matrix[u, v])
            w = math.exp(-d / sigma)
            if w > best:
                best = w
        scores[v] = float(p[v]) * float(best)
    return scores


def _select_by_threshold_floor_cap(
    nbrs: List[int],
    scores: Dict[int, float],
    tau: float,
    floor_k: int,
    cap_k: int,
) -> List[int]:
    ranked = sorted(nbrs, key=lambda v: scores.get(v, 0.0), reverse=True)

    chosen = [v for v in ranked if scores.get(v, 0.0) >= tau]

    if len(chosen) < int(floor_k):
        need = int(floor_k) - len(chosen)
        for v in ranked:
            if v not in chosen:
                chosen.append(v)
                need -= 1
                if need <= 0:
                    break

    if len(chosen) > int(cap_k):
        chosen = chosen[:int(cap_k)]

    return chosen


def _induced_subgraph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    nodes: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    nodes_sorted = list(nodes)
    node_set = set(nodes_sorted)
    mapping = {old: new for new, old in enumerate(nodes_sorted)}

    if edge_index.numel() == 0:
        sub_ei = torch.empty((2, 0), dtype=torch.long)
        sub_ea = torch.empty((0, edge_attr.shape[1]), dtype=torch.float32)
        return sub_ei, sub_ea, mapping

    keep_src, keep_dst, keep_attr = [], [], []
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for k, (u, v) in enumerate(zip(src, dst)):
        if (u in node_set) and (v in node_set):
            keep_src.append(mapping[u])
            keep_dst.append(mapping[v])
            keep_attr.append(edge_attr[k].tolist())

    if len(keep_src) == 0:
        sub_ei = torch.empty((2, 0), dtype=torch.long)
        sub_ea = torch.empty((0, edge_attr.shape[1]), dtype=torch.float32)
    else:
        sub_ei = torch.tensor([keep_src, keep_dst], dtype=torch.long)
        sub_ea = torch.tensor(keep_attr, dtype=torch.float32)

    return sub_ei, sub_ea, mapping

def build_pocket_residue_subgraphs(
    seq: str,
    binding_probs: List[float],
    pdb_path: str,
    chain_id: str = "A",
    dist_cutoff: float = 8.0,
    fully_connected: bool = False,

    rho_min: float = 0.55,
    rho_max: float = 0.85,
    k_min_frac: float = 0.01,
    k_max_frac: float = 0.25,
    k_min_abs: int = 5,

    eta: float = 0.85,
    exclude_other_components_in_expand: bool = True,

    hop_k: int = 1,
    lambda_thr: float = 0.30,

    alpha: float = 0.60,
    mmin: int = 3,
    mmax: int = 10,

    beta: float = 1.00,
    amin: int = 5,
    amax: int = 40,

    sigma_min: float = 2.0,
    sigma_max: float = 5.0,
    sigma_default: float = 3.0,

    # ---- output control ----
    return_global_graph: bool = False,
) -> Dict[str, Any]:
    """
    Build multi-pocket residue subgraphs using contact graph components.

    Returns dict:
      {
        'L': L,
        'c','rho','K','pK',
        'candidate_nodes': [...],
        'kept_components': [C1,C2,...],
        'subgraphs': [
           {
             'core_nodes': C,
             'added_nodes': added,
             'nodes': nodes_all,
             'sigma','tau','floor_k','cap_k',
             'edge_index','edge_attr',
             'node_id_map': mapping,
             'bridge_nodes': ...,
             'dropped_added': ...
           }, ...
        ],
        optional global graph fields
      }
    """
    edge_index, edge_attr, dist_matrix = build_residue_graph(
        seq=seq,
        pdb_path=pdb_path,
        chain_id=chain_id,
        dist_cutoff=dist_cutoff,
        fully_connected=fully_connected,
    )

    L = min(len(seq), len(binding_probs), int(dist_matrix.shape[0]))
    p = np.array(binding_probs[:L], dtype=np.float32)
    p = np.clip(p, 0.0, None)

    c = _entropy_concentration(p)
    rho = _rho_from_c(c, rho_min, rho_max)
    K0 = _k_mass_coverage(p, rho)
    K_knee = _k_knee_lower_bound(p)

    minK = max(int(k_min_abs), int(math.ceil(k_min_frac * L)))
    maxK = max(minK, int(math.ceil(k_max_frac * L)))
    K = max(K0, K_knee, minK)
    K = _clip_int(int(K), int(minK), int(maxK))

    topk = _topk_indices(p, K)
    cand_nodes = set(int(i) for i in topk.tolist())

    p_sorted = np.sort(p)[::-1]
    pK = float(p_sorted[K - 1]) if (K - 1) < len(p_sorted) else float(p_sorted[-1])

    contact_adj = _build_undirected_adj_from_edge_index(edge_index, L)

    comps = _connected_components_on_candidates_contact(cand_nodes=cand_nodes, contact_adj=contact_adj)

    kept_comps = _select_components_by_eta(comps=comps, p=p, eta=float(eta), topk_nodes=cand_nodes)

    kept_nodes_union: Set[int] = set()
    for C in kept_comps:
        kept_nodes_union |= set(C)
    tau = float(lambda_thr) * float(pK)
    subgraphs = []

    for C in kept_comps:
        Cset = set(C)

        reach = _bfs_hops(contact_adj, Cset, max_hop=int(hop_k))
        nbrs = sorted(list(reach - Cset))

        if exclude_other_components_in_expand:
            other_core = (kept_nodes_union - Cset)
            nbrs = [v for v in nbrs if v not in other_core]

        sigma = _sigma_for_component(
            C=C,
            nbrs=nbrs,
            dist_matrix=dist_matrix[:L, :L],
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            sigma_default=float(sigma_default),
        )

        scores = _score_neighbors(
            C=C,
            nbrs=nbrs,
            p=p,
            dist_matrix=dist_matrix[:L, :L],
            sigma=float(sigma),
        )

        floor_k = _clip_int(int(math.ceil(float(alpha) * len(C))), int(mmin), int(mmax))
        cap_k   = _clip_int(int(math.ceil(float(beta)  * len(C))), int(amin), int(amax))

        added = _select_by_threshold_floor_cap(
            nbrs=nbrs,
            scores=scores,
            tau=float(tau),
            floor_k=int(floor_k),
            cap_k=int(cap_k),
        )

        other_core = (kept_nodes_union - Cset)
        blocked = other_core if exclude_other_components_in_expand else set()

        kept_added, bridge_nodes, dropped_added = _enforce_connected_by_bridging(
            core_set=Cset,
            added=added,
            contact_adj=contact_adj,
            blocked=blocked,
            max_depth=int(hop_k),
        )
        added = kept_added

        nodes_all = sorted(list(Cset | set(added) | set(bridge_nodes)))

        sub_ei, sub_ea, mapping = _induced_subgraph(
            edge_index=edge_index,
            edge_attr=edge_attr,
            nodes=nodes_all,
        )

        subgraphs.append({
            "core_nodes": C,
            "added_nodes": added,
            "nodes": nodes_all,
            "sigma": float(sigma),
            "tau": float(tau),
            "floor_k": int(floor_k),
            "cap_k": int(cap_k),
            "edge_index": sub_ei,
            "edge_attr": sub_ea,
            "node_id_map": mapping,
            "bridge_nodes": bridge_nodes,
            "dropped_added": dropped_added,
        })

    out: Dict[str, Any] = {
        "L": int(L),
        "c": float(c),
        "rho": float(rho),
        "K": int(K),
        "pK": float(pK),
        "candidate_nodes": sorted(list(cand_nodes)),
        "kept_components": kept_comps,
        "subgraphs": subgraphs,
    }

    if return_global_graph:
        out.update({
            "global_edge_index": edge_index,
            "global_edge_attr": edge_attr,
            "dist_matrix": dist_matrix,
        })

    return out


def build_pocket_residue_subgraphs_precomputed(
    seq: str,
    binding_probs: List[float],

    global_edge_index: torch.Tensor,
    global_edge_attr: torch.Tensor,
    dist_matrix: np.ndarray,

    rho_min: float = 0.55,
    rho_max: float = 0.85,
    k_min_frac: float = 0.01,
    k_max_frac: float = 0.25,
    k_min_abs: int = 5,

    eta: float = 0.85,
    exclude_other_components_in_expand: bool = True,

    hop_k: int = 1,
    lambda_thr: float = 0.30,

    alpha: float = 0.60,
    mmin: int = 3,
    mmax: int = 10,

    beta: float = 1.00,
    amin: int = 5,
    amax: int = 40,

    sigma_min: float = 2.0,
    sigma_max: float = 5.0,
    sigma_default: float = 3.0,

    # ---- output control ----
    return_global_graph: bool = False,
) -> Dict[str, Any]:
    L = min(len(seq), len(binding_probs), int(dist_matrix.shape[0]))
    p = np.array(binding_probs[:L], dtype=np.float32)
    p = np.clip(p, 0.0, None)

    c = _entropy_concentration(p)
    rho = _rho_from_c(c, rho_min, rho_max)
    K0 = _k_mass_coverage(p, rho)
    K_knee = _k_knee_lower_bound(p)

    minK = max(int(k_min_abs), int(math.ceil(k_min_frac * L)))
    maxK = max(minK, int(math.ceil(k_max_frac * L)))
    K = max(K0, K_knee, minK)
    K = _clip_int(int(K), int(minK), int(maxK))

    topk = _topk_indices(p, K)
    cand_nodes = set(int(i) for i in topk.tolist())

    p_sorted = np.sort(p)[::-1]
    pK = float(p_sorted[K - 1]) if (K - 1) < len(p_sorted) else float(p_sorted[-1])

    contact_adj = _build_undirected_adj_from_edge_index(global_edge_index, L)

    comps = _connected_components_on_candidates_contact(cand_nodes=cand_nodes, contact_adj=contact_adj)

    kept_comps = _select_components_by_eta(comps=comps, p=p, eta=float(eta), topk_nodes=cand_nodes)

    kept_nodes_union: Set[int] = set()
    for C in kept_comps:
        kept_nodes_union |= set(C)

    tau = float(lambda_thr) * float(pK)
    dm = dist_matrix[:L, :L]

    subgraphs = []
    for C in kept_comps:
        Cset = set(C)

        reach = _bfs_hops(contact_adj, Cset, max_hop=int(hop_k))
        nbrs = sorted(list(reach - Cset))

        if exclude_other_components_in_expand:
            other_core = (kept_nodes_union - Cset)
            nbrs = [v for v in nbrs if v not in other_core]

        sigma = _sigma_for_component(
            C=C, nbrs=nbrs, dist_matrix=dm,
            sigma_min=float(sigma_min), sigma_max=float(sigma_max), sigma_default=float(sigma_default)
        )

        scores = _score_neighbors(C=C, nbrs=nbrs, p=p, dist_matrix=dm, sigma=float(sigma))

        floor_k = _clip_int(int(math.ceil(float(alpha) * len(C))), int(mmin), int(mmax))
        cap_k   = _clip_int(int(math.ceil(float(beta)  * len(C))), int(amin), int(amax))

        added = _select_by_threshold_floor_cap(
            nbrs=nbrs, scores=scores, tau=float(tau), floor_k=int(floor_k), cap_k=int(cap_k)
        )

        other_core = (kept_nodes_union - Cset)
        blocked = other_core if exclude_other_components_in_expand else set()

        kept_added, bridge_nodes, dropped_added = _enforce_connected_by_bridging(
            core_set=Cset,
            added=added,
            contact_adj=contact_adj,
            blocked=blocked,
            max_depth=int(hop_k),
        )
        added = kept_added

        nodes_all = sorted(list(Cset | set(added) | set(bridge_nodes)))

        sub_ei, sub_ea, mapping = _induced_subgraph(
            edge_index=global_edge_index,
            edge_attr=global_edge_attr,
            nodes=nodes_all,
        )

        subgraphs.append({
            "core_nodes": C,
            "added_nodes": added,
            "nodes": nodes_all,
            "sigma": float(sigma),
            "tau": float(tau),
            "floor_k": int(floor_k),
            "cap_k": int(cap_k),
            "edge_index": sub_ei,
            "edge_attr": sub_ea,
            "node_id_map": mapping,
            "bridge_nodes": bridge_nodes,
            "dropped_added": dropped_added,
        })

    out: Dict[str, Any] = {
        "L": int(L),
        "c": float(c),
        "rho": float(rho),
        "K": int(K),
        "pK": float(pK),
        "candidate_nodes": sorted(list(cand_nodes)),
        "kept_components": kept_comps,
        "subgraphs": subgraphs,
    }

    if return_global_graph:
        out.update({
            "global_edge_index": global_edge_index,
            "global_edge_attr": global_edge_attr,
            "dist_matrix": dist_matrix,
        })

    return out




