# precompute_esm2_chunked.py
# 作用：用 ESM-2 (esm2_t33_650M_UR50D) 为 proteins.txt 里的每条序列预计算 embedding，
#      支持超长序列的“滑动窗口 + overlap 重叠分块”，并把重叠区域按位置平均拼回全长 per-residue。
# 输出：dataset/davis/Esm2/<protein_id>.pt
# 说明：默认 SAVE_PER_RESIDUE=True，会保存 (L,1280) 的 per-residue（大文件）；同时也保存 per-sequence(1280,)

import os
import json
import re
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
import esm  # pip install fair-esm


PROTEIN_JSON = "dataset/kiba/proteins.txt"    # protein_id -> sequence 的 json
SAVE_DIR = "dataset/kiba/Esm2"               # 输出目录
MODEL_NAME = "esm2_t33_650M_UR50D"            # ESM2 650M (33-layer)
LOCAL_MODEL_PT = "/home/xwt/DPI/DPI/esm2_t33_650M_UR50D.pt"  # 本地权重

# ====== 分块参数 ======
MAX_RES = 1022            # 每个窗口最大 residues（不含BOS/EOS）
OVERLAP = 256             # 窗口重叠（128~512）
BATCH_SIZE = 2            # “窗口batch”，不是“蛋白batch”；650M 视显存调小/调大
SAVE_PER_RESIDUE = True   # True保存全长 per-residue (L,1280)，文件很大；False只保存 per-sequence
DTYPE_ON_CPU = torch.float32
USE_AMP = True

SKIP_EXISTING = True


def load_proteins(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(sorted(data.items(), key=lambda kv: kv[0]))


def safe_filename(pid: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._\-()\[\]]+", "_", pid)


def make_windows(seq: str, max_len: int, overlap: int) -> List[Tuple[int, str]]:
    L = len(seq)
    if L <= max_len:
        return [(0, seq)]

    step = max_len - overlap
    if step <= 0:
        raise ValueError(f"OVERLAP={overlap} too large for MAX_RES={max_len} (step<=0)")

    windows: List[Tuple[int, str]] = []
    start = 0
    while start < L:
        end = min(start + max_len, L)
        windows.append((start, seq[start:end]))
        if end == L:
            break
        start += step
    return windows


def ensure_hub_cache_has_weights(local_pt: str, model_name: str) -> None:
    if not local_pt or not os.path.isfile(local_pt):
        return

    hub_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(hub_dir, exist_ok=True)

    target = os.path.join(hub_dir, f"{model_name}.pt")
    if os.path.isfile(target):
        return

    try:
        import shutil
        shutil.copy2(local_pt, target)
        print(f"[OK] Copied local weight to hub cache: {target}")
    except Exception as e:
        print(f"[WARN] Failed to copy weight to hub cache. You can copy manually.\nReason: {repr(e)}")


def load_esm2_model(
    model_name: str,
    device: torch.device,
    local_pt: Optional[str] = None,
):
    if local_pt and os.path.isfile(local_pt):
        try:
            from esm.pretrained import load_model_and_alphabet_local
            model, alphabet = load_model_and_alphabet_local(local_pt)
            model.eval().to(device)
            print(f"[OK] Loaded ESM2 from local: {local_pt}")
            return model, alphabet
        except Exception as e:
            print(f"[WARN] Local load not available or failed, fallback to hub cache. Reason: {repr(e)}")

    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval().to(device)
    print("[OK] Loaded ESM2 via esm.pretrained (hub cache if available).")
    return model, alphabet


@torch.no_grad()
def embed_windows_batch(
    model,
    batch_converter,
    windows: List[Tuple[str, str]],   # List[(win_id, subseq)]
    repr_layer: int,
    device: torch.device,
    use_amp: bool = True,
):
    _, _, tokens = batch_converter([(wid, s) for wid, s in windows])
    tokens = tokens.to(device)

    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    else:
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)

    reps = out["representations"][repr_layer]  # [B, T, C]
    out_list = []
    for i, (wid, s) in enumerate(windows):
        L = len(s)
        per_res = reps[i, 1: L + 1].detach().cpu()  # [L, C]
        out_list.append((wid, per_res))
    return out_list


@torch.no_grad()
def embed_long_sequence_stitch(
    model,
    alphabet,
    seq: str,
    repr_layer: int,
    device: torch.device,
    max_len: int,
    overlap: int,
    batch_size: int,
    use_amp: bool = True,
):
    batch_converter = alphabet.get_batch_converter()
    windows = make_windows(seq, max_len=max_len, overlap=overlap)

    dim = getattr(model, "embed_dim", None)
    if dim is None:
        tmp = seq[: min(len(seq), max_len)]
        _, _, tok = batch_converter([("tmp", tmp)])
        tok = tok.to(device)
        out = model(tok, repr_layers=[repr_layer], return_contacts=False)
        dim = out["representations"][repr_layer].shape[-1]

    L = len(seq)
    sum_embed = torch.zeros((L, dim), dtype=torch.float32)
    cnt_embed = torch.zeros((L, 1), dtype=torch.float32)

    win_items = []
    for k, (start, sub) in enumerate(windows):
        win_id = f"win{k}"
        win_items.append((win_id, start, sub))

    for i in range(0, len(win_items), batch_size):
        batch = win_items[i: i + batch_size]
        batch_in = [(wid, sub) for (wid, _, sub) in batch]

        batch_out = embed_windows_batch(
            model=model,
            batch_converter=batch_converter,
            windows=batch_in,
            repr_layer=repr_layer,
            device=device,
            use_amp=use_amp,
        )
        wid2emb = {wid: emb for wid, emb in batch_out}

        for wid, start, _sub in batch:
            emb = wid2emb[wid]  # [len(sub), dim]
            sl = slice(start, start + emb.shape[0])
            sum_embed[sl] += emb.float()
            cnt_embed[sl] += 1.0

    full_per_residue = sum_embed / torch.clamp(cnt_embed, min=1.0)
    full_per_sequence = full_per_residue.mean(dim=0)
    return full_per_residue, full_per_sequence


@torch.no_grad()
def embed_long_sequence_chunk_mean(
    model,
    alphabet,
    seq: str,
    repr_layer: int,
    device: torch.device,
    max_len: int,
    overlap: int,
    batch_size: int,
    use_amp: bool = True,
):
    batch_converter = alphabet.get_batch_converter()
    windows = make_windows(seq, max_len=max_len, overlap=overlap)

    win_items = []
    for k, (_start, sub) in enumerate(windows):
        win_id = f"win{k}"
        win_items.append((win_id, len(sub), sub))

    chunk_vecs = []
    chunk_lens = []

    for i in range(0, len(win_items), batch_size):
        batch = win_items[i: i + batch_size]
        batch_in = [(wid, sub) for (wid, _, sub) in batch]

        batch_out = embed_windows_batch(
            model=model,
            batch_converter=batch_converter,
            windows=batch_in,
            repr_layer=repr_layer,
            device=device,
            use_amp=use_amp,
        )
        wid2emb = {wid: emb for wid, emb in batch_out}

        for wid, sub_len, _sub in batch:
            per_res = wid2emb[wid]          # [sub_len, dim]
            vec = per_res.mean(dim=0)       # [dim]
            chunk_vecs.append(vec)
            chunk_lens.append(sub_len)

    lens = torch.tensor(chunk_lens, dtype=torch.float32)
    mat = torch.stack(chunk_vecs, dim=0).float()  # [n, dim]
    full_vec = (mat * lens[:, None]).sum(dim=0) / lens.sum()
    return full_vec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    ensure_hub_cache_has_weights(LOCAL_MODEL_PT, MODEL_NAME)

    prot_dict = load_proteins(PROTEIN_JSON)
    print(f"Loaded {len(prot_dict)} proteins from {PROTEIN_JSON}")

    print(f"Loading model: {MODEL_NAME}")
    model, alphabet = load_esm2_model(MODEL_NAME, device=device, local_pt=LOCAL_MODEL_PT)
    print("model device:", next(model.parameters()).device)

    repr_layer = model.num_layers  # ESM2 取最后一层；t33=33
    print("repr_layer:", repr_layer)

    for pid, seq in tqdm(prot_dict.items(), desc="Embedding proteins"):
        seq = seq.upper().strip()
        if len(seq) == 0:
            continue

        out_name = safe_filename(pid) + ".pt"
        out_path = os.path.join(SAVE_DIR, out_name)

        if SKIP_EXISTING and os.path.exists(out_path):
            continue

        L = len(seq)

        if SAVE_PER_RESIDUE:
            per_res, per_seq = embed_long_sequence_stitch(
                model=model,
                alphabet=alphabet,
                seq=seq,
                repr_layer=repr_layer,
                device=device,
                max_len=MAX_RES,
                overlap=OVERLAP,
                batch_size=BATCH_SIZE,
                use_amp=USE_AMP,
            )

            per_res = per_res.to(DTYPE_ON_CPU)
            per_seq = per_seq.to(DTYPE_ON_CPU)

            torch.save(
                {
                    "id": pid,
                    "seq": seq,
                    "length": int(L),
                    "model": MODEL_NAME,
                    "repr_layer": int(repr_layer),
                    "max_res_window": int(MAX_RES),
                    "overlap": int(OVERLAP),
                    "pooling": "stitch_overlap_mean + mean_pool",
                    "dtype_saved": str(DTYPE_ON_CPU),
                    "per_residue": per_res,     # (L, 1280)
                    "per_sequence": per_seq,    # (1280,)
                },
                out_path,
            )
        else:
            per_seq = embed_long_sequence_chunk_mean(
                model=model,
                alphabet=alphabet,
                seq=seq,
                repr_layer=repr_layer,
                device=device,
                max_len=MAX_RES,
                overlap=OVERLAP,
                batch_size=BATCH_SIZE,
                use_amp=USE_AMP,
            ).to(DTYPE_ON_CPU)

            torch.save(
                {
                    "id": pid,
                    "seq": seq,
                    "length": int(L),
                    "model": MODEL_NAME,
                    "repr_layer": int(repr_layer),
                    "max_res_window": int(MAX_RES),
                    "overlap": int(OVERLAP),
                    "pooling": "chunk_mean(length_weighted)",
                    "dtype_saved": str(DTYPE_ON_CPU),
                    "per_sequence": per_seq,  # (1280,)
                },
                out_path,
            )

    print("Done! Saved to:", SAVE_DIR)


if __name__ == "__main__":
    print("torch.cuda.is_available():", torch.cuda.is_available())
    main()
