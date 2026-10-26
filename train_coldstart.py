import os
import ast
import json
import random
import math
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

import residue_graph as rg
from single_mol_atom_graph import build_atom_graph_from_smiles
from model import DPIMultiViewModel
from metrics import regression_metrics
from losses import hsic_loss, mse_loss_fn
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

ACTIVE_DATASET = "kiba"  # "davis" / "kiba"

# 可选：
# "warm_start"
# "drug_coldstart"
# "target_coldstart"
EXPERIMENT_SETTING = "target_coldstart_sampleval"

USE_PREDEFINED_SPLIT = True

SPLIT_DIR_MAP = {
    # warm-start splits
    ("davis", "warm_start"): os.path.join(
        ROOT, "splits", "davis_label_stratified_seed0"
    ),
    ("kiba", "warm_start"): os.path.join(
        ROOT, "splits", "kiba_label_stratified_seed0"
    ),

    ("davis", "drug_coldstart_sampleval"): os.path.join(
        ROOT, "splits", "davis_drug_coldstart_sampleval_multibal_seed0"
    ),
    ("davis", "target_coldstart_sampleval"): os.path.join(
        ROOT, "splits", "davis_target_coldstart_sampleval_multibal_seed0"
    ),
    ("kiba", "drug_coldstart_sampleval"): os.path.join(
        ROOT, "splits", "kiba_drug_coldstart_sampleval_multibal_seed0"
    ),
    ("kiba", "target_coldstart_sampleval"): os.path.join(
        ROOT, "splits", "kiba_target_coldstart_sampleval_multibal_seed0"
    ),
}

SPLIT_DIR = SPLIT_DIR_MAP[(ACTIVE_DATASET, EXPERIMENT_SETTING)]

DATASETS = {
    "davis": {
        "label_csv": os.path.join(ROOT, "dataset", "davis", "updated_full_with_seq_ligid.csv"),
        "prob_csv": os.path.join(ROOT, "dataset", "davis", "RESULT_all.csv"),
        "pdb_dir": os.path.join(ROOT, "dataset", "davis", "PDB"),
        "pdb_template": "{protein}.pdb",
        "chain_id": "A",
        "precomp_dir": os.path.join(ROOT, "precomp_residue", "davis"),
        "esm2_dir": os.path.join(ROOT, "dataset", "davis", "Esm2"),
    },
    "kiba": {
        "label_csv": os.path.join(ROOT, "dataset", "kiba", "updated_full_with_seq_ligid.csv"),
        "prob_csv": os.path.join(ROOT, "dataset", "kiba", "RESULT_all.csv"),
        "pdb_dir": os.path.join(ROOT, "dataset", "kiba", "PDB"),
        "pdb_template": "{protein}.pdb",
        "chain_id": "A",
        "precomp_dir": os.path.join(ROOT, "precomp_residue", "kiba"),
        "esm2_dir": None,
    },
}

CFG = DATASETS[ACTIVE_DATASET]

print(
    "[AMPICIP-v39] Full model: AMPIS + DUCGA + CIP + original GGTF "
    "(z_protein = z_res + gate * z_esm when ESM2 is available)."
)

SEED = 0
N_SPLITS = 5
VAL_RATIO_IN_TRAIN = 0.1

RUN_FOLDS = [1, 2, 3, 4, 5]

EPOCHS = 700
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 192
GAT_HEADS = 4
DROPOUT = 0.1

DIST_CUTOFF = 8.0
FULLY_CONNECTED_RES_GRAPH = False
ETA = 0.85

GRAD_CLIP = 5.0
USE_AMP = True
EARLY_STOP_PATIENCE = 100

TOP_K_BEST = 5

TRAIN_NUM_WORKERS = 6
EVAL_NUM_WORKERS = 0
NUM_WORKERS = EVAL_NUM_WORKERS

WARMUP_EPOCHS = 10
MIN_LR_RATIO = 0.01

CUDA_DEBUG_SYNC = False

exp_suffix = "v39_full"

SAVE_DIR = os.path.join(
    ROOT,
    "checkpoints_kfold",
    f"{ACTIVE_DATASET}_{EXPERIMENT_SETTING}_{exp_suffix}_h{HIDDEN_DIM}_seed{SEED}",
)

SKIP_FINISHED_FOLDS = True
RESUME_EARLY_STOPPED_FOLD_TO_EVAL = True

HSIC_MAX_LAMBDA = 1e-3
HSIC_WARMUP_EPOCHS = 50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_prob_list(x) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.tolist()]
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [float(v) for v in obj]
    except Exception:
        pass
    s2 = s.strip("[](){}")
    parts = [p.strip() for p in s2.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    return out

def assert_no_entity_leakage(df_train, df_val, df_test, setting: str):
    """
    strictval:
      train/val/test entity 全不重叠

    sampleval:
      train-val entity 可以重叠
      train-test entity 必须不重叠
      val-test entity 必须不重叠
    """
    if setting in ["drug_coldstart_strictval", "drug_coldstart_sampleval"]:
        entity_col = "ligand_id"
    elif setting in ["target_coldstart_strictval", "target_coldstart_sampleval"]:
        entity_col = "protein"
    elif setting == "warm_start":
        return
    else:
        raise ValueError(f"Unknown EXPERIMENT_SETTING: {setting}")

    allow_train_val_overlap = setting.endswith("_sampleval")

    train_entities = set(df_train[entity_col].astype(str))
    val_entities = set(df_val[entity_col].astype(str))
    test_entities = set(df_test[entity_col].astype(str))

    tv = train_entities & val_entities
    tt = train_entities & test_entities
    vt = val_entities & test_entities

    if len(tt) > 0 or len(vt) > 0:
        raise RuntimeError(
            f"[Leakage] setting={setting}, entity_col={entity_col}, "
            f"train-test overlap={len(tt)}, val-test overlap={len(vt)}"
        )

    if (not allow_train_val_overlap) and len(tv) > 0:
        raise RuntimeError(
            f"[Leakage] setting={setting}, entity_col={entity_col}, "
            f"strictval should have train-val overlap=0, but got {len(tv)}"
        )

    print(
        f"[ColdStartCheck] setting={setting}, entity_col={entity_col}: "
        f"train-val overlap={len(tv)}, "
        f"train-test overlap={len(tt)}, "
        f"val-test overlap={len(vt)}, "
        f"allow_train_val_overlap={allow_train_val_overlap}"
    )

def safe_filename(pid: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._\-()\[\]]+", "_", pid)

def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.3f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.3f}K"
    else:
        return str(n)


def print_parameter_summary(model, prefix: str = "[Model]"):
    stats = count_parameters(model)

    print(
        f"{prefix} total_params={stats['total_params']:,} "
        f"({format_params(stats['total_params'])}) | "
        f"trainable_params={stats['trainable_params']:,} "
        f"({format_params(stats['trainable_params'])}) | "
        f"non_trainable_params={stats['non_trainable_params']:,} "
        f"({format_params(stats['non_trainable_params'])})"
    )

def make_folds(indices: np.ndarray, n_splits: int) -> List[np.ndarray]:
    folds = []
    n = len(indices)
    fold_sizes = [n // n_splits + (1 if i < (n % n_splits) else 0) for i in range(n_splits)]
    cur = 0
    for fs in range(n_splits):
        size = fold_sizes[fs]
        folds.append(indices[cur:cur + size])
        cur += size
    return folds


def split_train_val(train_idx: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = train_idx.copy()
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_ratio))
    n_val = max(1, n_val) if len(idx) > 5 else max(0, n_val)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx


def induce_subgraph_from_global(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    nodes: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_sorted = list(nodes)
    node_set = set(nodes_sorted)
    mapping = {old: new for new, old in enumerate(nodes_sorted)}

    if edge_index is None or edge_index.numel() == 0:
        sub_ei = torch.empty((2, 0), dtype=torch.long)
        sub_ea = torch.empty((0, 2), dtype=torch.float32)
        return sub_ei, sub_ea

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    keep_src, keep_dst, keep_attr = [], [], []
    for k, (u, v) in enumerate(zip(src, dst)):
        u = int(u)
        v = int(v)
        if (u in node_set) and (v in node_set):
            keep_src.append(mapping[u])
            keep_dst.append(mapping[v])
            if edge_attr is not None and edge_attr.numel() > 0:
                keep_attr.append(edge_attr[k].tolist())
            else:
                keep_attr.append([0.0, 0.0])

    if len(keep_src) == 0:
        sub_ei = torch.empty((2, 0), dtype=torch.long)
        sub_ea = torch.empty((0, 2), dtype=torch.float32)
    else:
        sub_ei = torch.tensor([keep_src, keep_dst], dtype=torch.long)
        sub_ea = torch.tensor(keep_attr, dtype=torch.float32)

    if sub_ea.ndim != 2 or sub_ea.shape[1] != 2:
        raise RuntimeError(f"[Residue] edge_attr dim expected 2, got {tuple(sub_ea.shape)}")

    return sub_ei, sub_ea


def print_regression_diag(y_true_np: np.ndarray, mse: float, prefix: str = "[Val]"):
    y_true_np = np.asarray(y_true_np, dtype=np.float64).reshape(-1)
    if y_true_np.size == 0:
        print(f"{prefix} y_range=(nan,nan) std=nan baseline_mse(var)=nan rmse=nan r2=nan")
        return
    var = float(np.var(y_true_np))
    rmse = float(np.sqrt(float(mse)))
    r2 = 1.0 - float(mse) / (var + 1e-12)
    print(
        f"{prefix} y_range=({y_true_np.min():.3f},{y_true_np.max():.3f}) "
        f"std={np.std(y_true_np):.3f} baseline_mse(var)={var:.4f} "
        f"rmse={rmse:.4f} r2={r2:.4f}"
    )

def get_gpu_memory_info(device):
    if device.type != "cuda":
        return {
            "gpu_allocated_gb": np.nan,
            "gpu_reserved_gb": np.nan,
            "gpu_max_allocated_gb": np.nan,
            "gpu_max_reserved_gb": np.nan,
            "gpu_free_gb": np.nan,
            "gpu_total_gb": np.nan,
        }

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free = free_bytes / (1024 ** 3)
    total = total_bytes / (1024 ** 3)

    return {
        "gpu_allocated_gb": float(allocated),
        "gpu_reserved_gb": float(reserved),
        "gpu_max_allocated_gb": float(max_allocated),
        "gpu_max_reserved_gb": float(max_reserved),
        "gpu_free_gb": float(free),
        "gpu_total_gb": float(total),
    }


def append_epoch_log(log_path: str, row: dict):
    df_row = pd.DataFrame([row])
    write_header = not os.path.isfile(log_path)
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)


class DPIPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        pdb_dir: str,
        pdb_template: str = "{protein}.pdb",
        chain_id: str = "A",
        dist_cutoff: float = 8.0,
        fully_connected_res_graph: bool = False,
        eta: float = 0.85,
        precomp_dir: str = None,
        esm2_dir: Optional[str] = None,
        protein_esm_dim: int = 1280,
    ):
        self.df = df.reset_index(drop=True)
        self.pdb_dir = pdb_dir
        self.pdb_template = pdb_template
        self.chain_id = chain_id
        self.dist_cutoff = float(dist_cutoff)
        self.fully_connected_res_graph = bool(fully_connected_res_graph)
        self.eta = float(eta)

        self.precomp_dir = precomp_dir
        if self.precomp_dir is not None:
            print(f"[Residue] Use precomputed global residue graphs from: {self.precomp_dir}")

        self._global_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, int]] = {}

        # ESM2
        self.esm2_dir = esm2_dir
        self.protein_esm_dim = int(protein_esm_dim)
        self.use_protein_esm = self.esm2_dir is not None
        if self.use_protein_esm:
            print(f"[ESM2] Use precomputed per-sequence ESM2 from: {self.esm2_dir}")
        self._esm_cache: Dict[str, Optional[torch.Tensor]] = {}

    def __len__(self):
        return len(self.df)

    def _resolve_pdb_path(self, protein: str, ligand_id: Any) -> str:
        fn = self.pdb_template.format(protein=protein, ligand_id=str(ligand_id))
        return os.path.join(self.pdb_dir, fn)

    def _get_global_res_graph(
        self,
        protein: str,
        seq: str,
        pdb_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, int]:
        if self.precomp_dir is not None:
            key = f"pre::{protein}"
            if key in self._global_cache:
                return self._global_cache[key]

            pre_path = os.path.join(self.precomp_dir, f"{protein}.pt")
            if os.path.isfile(pre_path):
                obj = torch.load(pre_path, map_location="cpu", weights_only=False)

                gei = obj["edge_index"].long()
                gea = obj["edge_attr"].float()

                dist_matrix = obj["dist_matrix"]
                if isinstance(dist_matrix, torch.Tensor):
                    dist_matrix = dist_matrix.cpu().numpy()

                x_full = obj["x_full"]
                if isinstance(x_full, np.ndarray):
                    x_full = torch.tensor(x_full, dtype=torch.float32)
                else:
                    x_full = x_full.float()

                L_dist = int(dist_matrix.shape[0])
                L_feat = int(x_full.shape[0])
                L_seq = int(len(seq))
                L_global = min(L_dist, L_feat, L_seq)

                if (L_dist != L_global) or (L_feat != L_global):
                    dist_matrix = dist_matrix[:L_global, :L_global]
                    x_full = x_full[:L_global]

                    mask = (gei[0] < L_global) & (gei[1] < L_global)
                    gei = gei[:, mask]
                    if gea.numel() > 0:
                        gea = gea[mask]

                self._global_cache[key] = (gei, gea, dist_matrix, x_full, L_global)
                return gei, gea, dist_matrix, x_full, L_global

            print(
                f"[Residue][WARN] precomputed file not found for protein={protein}, "
                f"fallback to on-the-fly PDB build."
            )

        key = f"pdb::{pdb_path}|{self.chain_id}|{self.dist_cutoff}|{int(self.fully_connected_res_graph)}"
        if key in self._global_cache:
            return self._global_cache[key]

        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        gei, gea, dist_matrix = rg.build_residue_graph(
            seq=seq,
            pdb_path=pdb_path,
            chain_id=self.chain_id,
            dist_cutoff=self.dist_cutoff,
            fully_connected=self.fully_connected_res_graph,
        )

        if hasattr(rg, "build_residue_features_from_pdb"):
            x_np = rg.build_residue_features_from_pdb(seq, pdb_path, chain_id=self.chain_id)
        else:
            x_np = rg.build_residue_features_from_sequence(seq)
        x_full = torch.tensor(x_np, dtype=torch.float32)

        L_dist = int(dist_matrix.shape[0])
        L_feat = int(x_full.shape[0])
        L_seq = int(len(seq))
        L_global = min(L_dist, L_feat, L_seq)

        if (L_dist != L_global) or (L_feat != L_global):
            dist_matrix = dist_matrix[:L_global, :L_global]
            x_full = x_full[:L_global]
            mask = (gei[0] < L_global) & (gei[1] < L_global)
            gei = gei[:, mask]
            if gea.numel() > 0:
                gea = gea[mask]

        self._global_cache[key] = (gei, gea, dist_matrix, x_full, L_global)
        return gei, gea, dist_matrix, x_full, L_global

    def _get_protein_esm(self, protein: str) -> Optional[torch.Tensor]:
        if not self.use_protein_esm:
            return None

        if protein in self._esm_cache:
            return self._esm_cache[protein]

        fn = safe_filename(protein) + ".pt"
        path = os.path.join(self.esm2_dir, fn)
        if not os.path.isfile(path):
            print(f"[ESM2][WARN] file not found for protein={protein}: {path}")
            self._esm_cache[protein] = None
            return None

        obj = torch.load(path, map_location="cpu", weights_only=False)
        if "per_sequence" not in obj:
            print(f"[ESM2][WARN] 'per_sequence' not in {path}")
            self._esm_cache[protein] = None
            return None

        vec = obj["per_sequence"]
        if isinstance(vec, torch.Tensor):
            vec = vec.float()
        else:
            vec = torch.tensor(vec, dtype=torch.float32)

        if vec.ndim != 1:
            vec = vec.view(-1)
        if vec.shape[0] != self.protein_esm_dim:
            print(
                f"[ESM2][WARN] dim mismatch for protein={protein}: "
                f"{vec.shape[0]} vs expected {self.protein_esm_dim}. "
                f"Will pad/crop."
            )
            if vec.shape[0] < self.protein_esm_dim:
                pad = self.protein_esm_dim - vec.shape[0]
                vec = torch.cat([vec, torch.zeros(pad, dtype=torch.float32)], dim=0)
            else:
                vec = vec[: self.protein_esm_dim]

        self._esm_cache[protein] = vec
        return vec

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        protein = str(row["protein"])
        seq = str(row["sequence"])
        smiles = str(row["ligand"])
        ligand_id = row.get("ligand_id", "")

        y = float(row["label"])
        probs = parse_prob_list(row["Binding Site Probability"])

        atom_data = build_atom_graph_from_smiles(smiles, explicit_H=True, use_chirality=False)

        pdb_path = self._resolve_pdb_path(protein, ligand_id)
        gei, gea, dist_matrix, x_full, L_global = self._get_global_res_graph(
            protein=protein,
            seq=seq,
            pdb_path=pdb_path,
        )

        if hasattr(rg, "build_pocket_residue_subgraphs_precomputed"):
            out = rg.build_pocket_residue_subgraphs_precomputed(
                seq=seq,
                binding_probs=probs,
                global_edge_index=gei,
                global_edge_attr=gea,
                dist_matrix=dist_matrix,
                eta=self.eta,
                return_global_graph=False,
            )
        else:
            out = rg.build_pocket_residue_subgraphs(
                seq=seq,
                binding_probs=probs,
                pdb_path=pdb_path,
                chain_id=self.chain_id,
                dist_cutoff=self.dist_cutoff,
                fully_connected=self.fully_connected_res_graph,
                eta=self.eta,
                return_global_graph=False,
            )

        L = int(out.get("L", L_global))
        L = int(min(L, L_global))
        x_full = x_full[:L]

        subgraphs = out.get("subgraphs", [])
        union_nodes = set()
        for sg in subgraphs:
            for n in sg.get("nodes", []):
                n = int(n)
                if 0 <= n < L:
                    union_nodes.add(n)

        if len(union_nodes) == 0:
            res_data = Data(
                x=torch.zeros((1, x_full.shape[1]), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 2), dtype=torch.float32),
            )
        else:
            union_nodes_sorted = sorted(list(union_nodes))
            sub_ei, sub_ea = induce_subgraph_from_global(gei, gea, union_nodes_sorted)
            x_union = x_full[union_nodes_sorted]
            res_data = Data(
                x=x_union,
                edge_index=sub_ei,
                edge_attr=sub_ea,
            )

        protein_esm = None
        if self.use_protein_esm:
            protein_esm = self._get_protein_esm(protein)
            if protein_esm is None:
                protein_esm = torch.zeros(self.protein_esm_dim, dtype=torch.float32)

        return atom_data, res_data, torch.tensor([y], dtype=torch.float32), protein_esm


def collate_fn(batch):
    atoms, ress, ys, esms = zip(*batch)
    atom_batch = Batch.from_data_list(list(atoms))
    res_batch = Batch.from_data_list(list(ress))
    y = torch.cat(list(ys), dim=0).view(-1)

    if esms[0] is None:
        protein_esm = None
    else:
        protein_esm = torch.stack(list(esms), dim=0)

    return atom_batch, res_batch, y, protein_esm


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_all, y_pred_all = [], []
    for atom_data, res_data, y, protein_esm in loader:
        atom_data = atom_data.to(device)
        res_data = res_data.to(device)
        y = y.to(device).view(-1)

        if isinstance(protein_esm, torch.Tensor):
            protein_esm = protein_esm.to(device)
        else:
            protein_esm = None

        out = model(atom_data, res_data, protein_esm=protein_esm)
        pred = out["logit"].view(-1)

        y_true_all.append(y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    y_true_all = torch.cat(y_true_all, dim=0).numpy()
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy()
    return y_true_all, y_pred_all


@torch.no_grad()
def evaluate(model, loader, device, return_arrays: bool = False):
    y_true, y_pred = predict(model, loader, device)
    met = regression_metrics(y_true, y_pred, with_spearman=True, with_ci=True)
    if return_arrays:
        return met, y_true, y_pred
    return met


def get_hsic_lambda(epoch: int) -> float:
    if epoch <= 150:
        return 1e-3
    elif epoch <= 350:
        t = (epoch - 150) / (350 - 150)  # [0,1]
        return 1e-3 - t * (1e-3 - 5e-4)
    else:
        return 5e-4



def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    lambda_t = get_hsic_lambda(epoch)

    for batch_idx, (atom_data, res_data, y, protein_esm) in enumerate(loader):
        atom_data = atom_data.to(device)
        res_data = res_data.to(device)
        y = y.to(device).view(-1)

        if isinstance(protein_esm, torch.Tensor):
            protein_esm = protein_esm.to(device)
        else:
            protein_esm = None

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                out = model(atom_data, res_data, protein_esm=protein_esm)
                pred = out["logit"].view(-1)

                mse = mse_loss_fn(pred, y)
                z_drug = out["z_drug"]
                z_protein = out["z_protein"]
                aux_hsic = hsic_loss(z_drug, z_protein)

                loss = mse + lambda_t * aux_hsic

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss detected: {loss.item()} -> skip this step")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                continue

            scaler.scale(loss).backward()

            if CUDA_DEBUG_SYNC and device.type == "cuda":
                torch.cuda.synchronize()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            out = model(atom_data, res_data, protein_esm=protein_esm)
            pred = out["logit"].view(-1)

            mse = mse_loss_fn(pred, y)
            z_drug = out["z_drug"]
            z_protein = out["z_protein"]
            aux_hsic = hsic_loss(z_drug, z_protein)

            loss = mse + lambda_t * aux_hsic

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss detected: {loss.item()} -> skip this step")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                continue

            loss.backward()

            if CUDA_DEBUG_SYNC and device.type == "cuda":
                torch.cuda.synchronize()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

        if batch_idx == 0:
            mse_val = float(mse.detach().cpu().item())
            hsic_val = float(aux_hsic.detach().cpu().item())
            print(
                f"    [Epoch {epoch:04d} | Batch {batch_idx:04d}] "
                f"mse={mse_val:.6f} hsic={hsic_val:.6f} "
                f"lambda={lambda_t:.6e} lambda*hsic={lambda_t * hsic_val:.6f}"
            )

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(1, n)

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_epochs: int,
    num_training_epochs: int,
    min_lr_ratio: float = 0.01,
):
    def lr_lambda(current_epoch: int):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch + 1) / float(max(1, num_warmup_epochs))

        progress = (current_epoch - num_warmup_epochs) / float(
            max(1, num_training_epochs - num_warmup_epochs)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def load_finished_fold_outputs(fold_dir: str):
    metrics_path = os.path.join(fold_dir, "metrics.json")
    pred_path = os.path.join(fold_dir, "test_predictions.csv")

    if (not os.path.isfile(metrics_path)) or (not os.path.isfile(pred_path)):
        return False, None, None, None

    with open(metrics_path, "r", encoding="utf-8") as f:
        met_obj = json.load(f)

    val_met = met_obj["val"]
    test_met = met_obj["test"]
    pred_df = pd.read_csv(pred_path)

    return True, val_met, test_met, pred_df

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    use_amp = bool(USE_AMP and device.type == "cuda")

    df_label = pd.read_csv(CFG["label_csv"])
    df_prob = pd.read_csv(CFG["prob_csv"])

    merge_cols = ["protein", "ligand", "ligand_id", "sequence"]
    df = pd.merge(df_label, df_prob, on=merge_cols, how="inner")
    if len(df) == 0:
        raise RuntimeError("Merge结果为空：请检查两个CSV的列名/内容是否一致。")

    df = df.reset_index(drop=True).copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    n_total = len(df)
    idx_all = np.arange(n_total)

    if not USE_PREDEFINED_SPLIT:
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx_all)
        folds = make_folds(idx_all, n_splits=N_SPLITS)
    else:
        folds = None

    fold_val_metrics = []
    fold_test_metrics = []

    oof_pred = np.full((n_total,), np.nan, dtype=np.float32)
    oof_true = df["label"].to_numpy(dtype=np.float32)

    for fold_id in range(N_SPLITS):
        fold_no = fold_id + 1
        if RUN_FOLDS is not None and fold_no not in RUN_FOLDS:
            print(f"[Fold {fold_no}] Skip because RUN_FOLDS={RUN_FOLDS}")
            continue

        if USE_PREDEFINED_SPLIT:
            fold_split_dir = os.path.join(SPLIT_DIR, f"fold{fold_no}")

            train_csv = os.path.join(fold_split_dir, "split_train.csv")
            val_csv = os.path.join(fold_split_dir, "split_val.csv")
            test_csv = os.path.join(fold_split_dir, "split_test.csv")

            if not os.path.isfile(train_csv):
                raise FileNotFoundError(f"Missing predefined train split: {train_csv}")
            if not os.path.isfile(val_csv):
                raise FileNotFoundError(f"Missing predefined val split: {val_csv}")
            if not os.path.isfile(test_csv):
                raise FileNotFoundError(f"Missing predefined test split: {test_csv}")

            df_train = pd.read_csv(train_csv)
            df_val = pd.read_csv(val_csv)
            df_test = pd.read_csv(test_csv)

            assert_no_entity_leakage(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                setting=EXPERIMENT_SETTING,
            )

        else:
            test_idx = folds[fold_id]
            train_all_idx = np.concatenate(
                [folds[i] for i in range(N_SPLITS) if i != fold_id]
            )

            train_idx, val_idx = split_train_val(
                train_all_idx,
                VAL_RATIO_IN_TRAIN,
                seed=SEED + 1000 + fold_id,
            )

            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

        fold_dir = os.path.join(SAVE_DIR, f"fold{fold_id + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        if SKIP_FINISHED_FOLDS:
            finished, val_met_done, test_met_done, pred_df_done = load_finished_fold_outputs(fold_dir)

            if finished:
                print(f"\n========== Fold {fold_id + 1}/{N_SPLITS} ==========")
                print(f"[Fold {fold_id + 1}] Already finished. Skip training/evaluation.")

                fold_val_metrics.append(val_met_done)
                fold_test_metrics.append(test_met_done)

                if ("row_id" in pred_df_done.columns) and ("pred_logit" in pred_df_done.columns):
                    row_ids = pred_df_done["row_id"].to_numpy(dtype=np.int64)
                    preds = pred_df_done["pred_logit"].to_numpy(dtype=np.float32)
                    oof_pred[row_ids] = preds

                print(
                    f"[Fold {fold_id + 1} Test] "
                    f"mse={test_met_done['mse']:.6f} "
                    f"spearman={test_met_done.get('spearman', float('nan')):.4f} "
                    f"pearson={test_met_done.get('pearson', float('nan')):.4f} "
                    f"ci={test_met_done.get('ci', float('nan')):.4f}"
                )
                continue

        df_train.to_csv(os.path.join(fold_dir, "split_train.csv"), index=False)
        df_val.to_csv(os.path.join(fold_dir, "split_val.csv"), index=False)
        df_test.to_csv(os.path.join(fold_dir, "split_test.csv"), index=False)

        print(f"\n========== Fold {fold_id + 1}/{N_SPLITS} ==========")
        print(
            f"[Split] setting={EXPERIMENT_SETTING} | "
            f"train={len(df_train)} val={len(df_val)} test={len(df_test)}"
        )
        print(f"[Save ] {fold_dir}/split_train.csv | split_val.csv | split_test.csv")

        common_ds_kwargs = dict(
            pdb_dir=CFG["pdb_dir"],
            pdb_template=CFG["pdb_template"],
            chain_id=CFG["chain_id"],
            dist_cutoff=DIST_CUTOFF,
            fully_connected_res_graph=FULLY_CONNECTED_RES_GRAPH,
            eta=ETA,
            precomp_dir=CFG.get("precomp_dir"),
            esm2_dir=CFG.get("esm2_dir"),
            protein_esm_dim=1280,
        )

        train_set = DPIPairDataset(df_train, **common_ds_kwargs)
        val_set = DPIPairDataset(df_val, **common_ds_kwargs)
        test_set = DPIPairDataset(df_test, **common_ds_kwargs)

        def seed_worker(worker_id):
            worker_seed = SEED + fold_id * 100 + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(SEED + fold_id)

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=TRAIN_NUM_WORKERS,
            collate_fn=collate_fn,
            drop_last=False,
            prefetch_factor=1 if TRAIN_NUM_WORKERS > 0 else None,
            persistent_workers=(TRAIN_NUM_WORKERS > 0),
            generator=g,
            worker_init_fn=seed_worker if TRAIN_NUM_WORKERS > 0 else None,
        )

        if EVAL_NUM_WORKERS > 0:
            val_loader = DataLoader(
                val_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=EVAL_NUM_WORKERS,
                collate_fn=collate_fn,
                drop_last=False,
                prefetch_factor=1,
                persistent_workers=False,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=EVAL_NUM_WORKERS,
                collate_fn=collate_fn,
                drop_last=False,
                prefetch_factor=1,
                persistent_workers=False,
            )
        else:
            val_loader = DataLoader(
                val_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                drop_last=False,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                drop_last=False,
            )

        model = DPIMultiViewModel(
            hidden_dim=HIDDEN_DIM,
            gat_heads=GAT_HEADS,
            dropout=DROPOUT,
            atom_in_dim=55,
            res_in_dim=47,
            atom_edge_dim=16,
            res_edge_dim=2,
            use_protein_esm=(CFG.get("esm2_dir") is not None),
        ).to(device)
        
        if fold_no == RUN_FOLDS[0]:
            print_parameter_summary(model, prefix="[Model]")

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            num_warmup_epochs=WARMUP_EPOCHS,
            num_training_epochs=EPOCHS,
            min_lr_ratio=MIN_LR_RATIO,
        )

        best_val_mse = float("inf")
        best_epoch = -1
        bad_count = 0

        best_path = os.path.join(fold_dir, "best.pt")
        last_path = os.path.join(fold_dir, "last.pt")
        topk_best = []

        start_epoch = 1
        skip_training_loop = False

        if os.path.isfile(last_path):
            ckpt_last = torch.load(last_path, map_location=device, weights_only=False)
            try:
                model.load_state_dict(ckpt_last["model"])
                optimizer.load_state_dict(ckpt_last["optimizer"])
                scheduler.load_state_dict(ckpt_last["scheduler"])

                best_val_mse = ckpt_last.get("best_val_mse", best_val_mse)
                best_epoch = ckpt_last.get("best_epoch", best_epoch)
                bad_count = ckpt_last.get("bad_count", bad_count)
                topk_best = ckpt_last.get("topk_best", topk_best)

                last_epoch = ckpt_last.get("epoch", 0)
                start_epoch = last_epoch + 1

                print(
                    f"[Fold {fold_id + 1}] Resume from epoch={last_epoch} "
                    f"(best_val_mse={best_val_mse:.6f}, best_epoch={best_epoch}, bad_count={bad_count})"
                )

                if RESUME_EARLY_STOPPED_FOLD_TO_EVAL and bad_count >= EARLY_STOP_PATIENCE:
                    print(
                        f"[Fold {fold_id + 1}] Last checkpoint already reached early stopping "
                        f"(bad_count={bad_count}, patience={EARLY_STOP_PATIENCE}). "
                        f"Skip training loop and go directly to final evaluation."
                    )
                    skip_training_loop = True

                if start_epoch > EPOCHS:
                    print(
                        f"[Fold {fold_id + 1}] start_epoch={start_epoch} > EPOCHS={EPOCHS}. "
                        f"Skip training loop and go directly to final evaluation."
                    )
                    skip_training_loop = True

            except Exception as e:
                print(f"[Fold {fold_id + 1}] WARN: failed to load last.pt for resume: {e}")
                start_epoch = 1
                skip_training_loop = False

        if not skip_training_loop:
            for epoch in range(start_epoch, EPOCHS + 1):
                epoch_start = time.time()

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    use_amp=use_amp,
                    grad_clip=GRAD_CLIP,
                    epoch=epoch,
                )

                val_met, y_val_np, _ = evaluate(model, val_loader, device, return_arrays=True)
                val_mse = float(val_met["mse"])

                if device.type == "cuda":
                    torch.cuda.synchronize()
                epoch_time = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]["lr"]
                current_lambda = get_hsic_lambda(epoch)

                gpu_info = get_gpu_memory_info(device)

                print(
                    f"[Fold {fold_id + 1} | Epoch {epoch:04d}] "
                    f"train_loss={train_loss:.6f} | "
                    f"val_mse={val_mse:.6f} "
                    f"val_spearman={val_met.get('spearman', float('nan')):.4f} "
                    f"val_pearson={val_met.get('pearson', float('nan')):.4f} "
                    f"val_ci={val_met.get('ci', float('nan')):.4f} | "
                    f"epoch_time={epoch_time:.2f}s ({epoch_time/60:.2f} min) | "
                    f"lr={current_lr:.6e} | "
                    f"lambda(epoch)={current_lambda:.6e} | "
                    f"gpu_alloc={gpu_info['gpu_allocated_gb']:.2f}GB "
                    f"gpu_reserved={gpu_info['gpu_reserved_gb']:.2f}GB "
                    f"gpu_max_alloc={gpu_info['gpu_max_allocated_gb']:.2f}GB "
                    f"gpu_max_reserved={gpu_info['gpu_max_reserved_gb']:.2f}GB "
                    f"gpu_free={gpu_info['gpu_free_gb']:.2f}/{gpu_info['gpu_total_gb']:.2f}GB"
                )
                print_regression_diag(y_val_np, val_mse, prefix="  [Val]")

                improved = False
                if val_mse < best_val_mse - 1e-12:
                    best_val_mse = val_mse
                    best_epoch = epoch
                    bad_count = 0
                    improved = True

                    ckpt = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_mse": best_val_mse,
                        "fold": fold_id + 1,
                        "active_dataset": ACTIVE_DATASET,
                        "cfg": CFG,
                        "train_cfg": {
                            "seed": SEED,
                            "n_splits": N_SPLITS,
                            "val_ratio_in_train": VAL_RATIO_IN_TRAIN,
                            "epochs": EPOCHS,
                            "batch_size": BATCH_SIZE,
                            "lr": LR,
                            "weight_decay": WEIGHT_DECAY,
                            "hidden_dim": HIDDEN_DIM,
                            "gat_heads": GAT_HEADS,
                            "dropout": DROPOUT,
                            "dist_cutoff": DIST_CUTOFF,
                            "fully_connected_res_graph": FULLY_CONNECTED_RES_GRAPH,
                            "eta": ETA,
                            "use_amp": use_amp,
                            "warmup_epochs": WARMUP_EPOCHS,
                            "min_lr_ratio": MIN_LR_RATIO,
                            "hsic_max_lambda": HSIC_MAX_LAMBDA,
                            "hsic_warmup_epochs": HSIC_WARMUP_EPOCHS,
                            "top_k_best": TOP_K_BEST,
                        },
                    }

                    torch.save(ckpt, best_path)

                    topk_path = os.path.join(fold_dir, f"best_epoch{epoch:04d}_mse{val_mse:.6f}.pt")
                    torch.save(ckpt, topk_path)

                    topk_best.append(
                        {
                            "epoch": epoch,
                            "val_mse": float(val_mse),
                            "path": topk_path,
                        }
                    )

                    topk_best = sorted(topk_best, key=lambda x: x["val_mse"])[:TOP_K_BEST]

                    keep_paths = set(item["path"] for item in topk_best)
                    for fn in os.listdir(fold_dir):
                        if fn.startswith("best_epoch") and fn.endswith(".pt"):
                            p = os.path.join(fold_dir, fn)
                            if p not in keep_paths:
                                try:
                                    os.remove(p)
                                except Exception as e:
                                    print(f"[WARN] failed to remove old top-k ckpt {p}: {e}")

                    print(
                        f"  ✅ best updated: val_mse={best_val_mse:.6f} "
                        f"@ epoch={epoch}, saved to {best_path}"
                    )
                    print(
                        f"  ✅ top-{TOP_K_BEST} updated: keep {[round(x['val_mse'], 6) for x in topk_best]}"
                    )
                else:
                    bad_count += 1

                early_stop_reached = bad_count >= EARLY_STOP_PATIENCE

                if not early_stop_reached:
                    scheduler.step()

                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_mse": best_val_mse,
                        "best_epoch": best_epoch,
                        "bad_count": bad_count,
                        "topk_best": topk_best,
                        "fold": fold_id + 1,
                        "active_dataset": ACTIVE_DATASET,
                        "cfg": CFG,
                        "early_stop_reached": bool(early_stop_reached),
                    },
                    last_path,
                )

                if early_stop_reached:
                    print(
                        f"  ⏹ Early stopping: val_mse not improved for "
                        f"{EARLY_STOP_PATIENCE} epochs. "
                        f"Best epoch={best_epoch}, best_val_mse={best_val_mse:.6f}"
                    )
                    break

        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(
                f"[Fold {fold_id + 1}] Load best: epoch={ckpt.get('epoch')} "
                f"best_val_mse={ckpt.get('best_val_mse')}"
            )
        else:
            print(f"[Fold {fold_id + 1}] WARN: best checkpoint not found, evaluate current model.")

        val_met, y_val_np, y_val_pred_np = evaluate(model, val_loader, device, return_arrays=True)
        test_met, y_test_np, y_test_pred_np = evaluate(model, test_loader, device, return_arrays=True)

        fold_val_metrics.append(val_met)
        fold_test_metrics.append(test_met)

        print(
            f"[Fold {fold_id + 1} Val ] mse={val_met['mse']:.6f} "
            f"spearman={val_met.get('spearman', float('nan')):.4f} "
            f"pearson={val_met.get('pearson', float('nan')):.4f} "
            f"ci={val_met.get('ci', float('nan')):.4f}"
        )
        print_regression_diag(y_val_np, float(val_met["mse"]), prefix="  [Val-final]")
        print(
            f"[Fold {fold_id + 1} Test] mse={test_met['mse']:.6f} "
            f"spearman={test_met.get('spearman', float('nan')):.4f} "
            f"pearson={test_met.get('pearson', float('nan')):.4f} "
            f"ci={test_met.get('ci', float('nan')):.4f}"
        )

        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"val": val_met, "test": test_met}, f, indent=2)

        global_row_ids = df_test["row_id"].to_numpy(dtype=np.int64)
        oof_pred[global_row_ids] = y_test_pred_np.astype(np.float32)

        pred_df = df_test[["row_id", "protein", "ligand_id", "label"]].copy()
        pred_df["pred_logit"] = y_test_pred_np.astype(np.float32)
        pred_df.to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)

        del train_loader, val_loader, test_loader
        del train_set, val_set, test_set
        del model, optimizer, scheduler, scaler

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def mean_std(metrics_list: List[Dict[str, float]], key: str) -> Tuple[float, float]:
        vals = np.array([m.get(key, np.nan) for m in metrics_list], dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    print("\n========== K-Fold Summary (fold-wise test) ==========")
    for k in ["mse", "spearman", "pearson", "ci"]:
        m, s = mean_std(fold_test_metrics, k)
        print(f"[Test] {k:8s}: mean={m:.6f} std={s:.6f}")

    ok = ~np.isnan(oof_pred)
    if ok.sum() == n_total:
        oof_met = regression_metrics(
            oof_true.astype(np.float32),
            oof_pred.astype(np.float32),
            with_spearman=True,
            with_ci=True,
        )
        print("\n========== OOF Overall (each sample predicted once) ==========")
        print(
            f"[OOF] mse={oof_met['mse']:.6f} "
            f"spearman={oof_met.get('spearman', float('nan')):.4f} "
            f"pearson={oof_met.get('pearson', float('nan')):.4f} "
            f"ci={oof_met.get('ci', float('nan')):.4f}"
        )

        oof_df = df[["row_id", "protein", "ligand_id", "label"]].copy()
        oof_df["oof_pred_logit"] = oof_pred
        oof_df.to_csv(os.path.join(SAVE_DIR, "oof_predictions_all.csv"), index=False)
        print(f"[Save] {os.path.join(SAVE_DIR, 'oof_predictions_all.csv')}")
    else:
        print(f"[WARN] OOF predictions not fully filled: filled={ok.sum()} / total={n_total}")

    print(f"\n[Done] All saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
