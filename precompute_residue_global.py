import os
import torch
import pandas as pd
import numpy as np

import residue_graph as rg
print("[DEBUG] residue_graph loaded from:", rg.__file__)


'''
构建残基图并进行保存.pt文件

先加载 precompute_residue_global.py 生成的每个蛋白 .pt（global residue graph + dist + x_full），
再在 residue_graph.py 里用 build_pocket_residue_subgraphs_precomputed() 根据 binding_probs 选 pocket 节点，
'''
'''
feat = np.concatenate(
    [
        f_type,   # 21
        f_hydro,  # 1
        f_charge, # 1
        f_polar,  # 1
        f_ss8,    # 8
        f_sasa,   # 1
        f_arom,   # 1
        f_vol,    # 1
        f_pl,     # 1
        f_h2,     # 1
        f_kid,    # 10
    ],
    axis=0
)
# 共 47 维
'''

ROOT = os.path.dirname(os.path.abspath(__file__))

PDB_DIR = os.path.join(ROOT, "dataset", "davis", "PDB")
LABEL_CSV = os.path.join(ROOT, "dataset", "davis", "updated_full_with_seq_ligid.csv")
OUT_DIR = os.path.join(ROOT, "precomp_residue", "davis")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(LABEL_CSV)

df_prot = df.drop_duplicates(subset=["protein", "sequence"])

bad_proteins_stats = []

for _, row in df_prot.iterrows():
    protein = str(row["protein"])
    seq = str(row["sequence"])
    pdb_path = os.path.join(PDB_DIR, f"{protein}.pdb")

    print("Precompute", protein)
    
    gei, gea, dist_matrix = rg.build_residue_graph(
    seq=seq,
    pdb_path=pdb_path,
    chain_id="A",
    dist_cutoff=8.0,
    fully_connected=False,
)
    print("[DEBUG]", protein, "gea.shape =", tuple(gea.shape))
    assert gea.ndim == 2 and gea.shape[1] == 17, f"{protein}: got edge_attr shape {tuple(gea.shape)}"


    gei, gea, dist_matrix = rg.build_residue_graph(
        seq=seq,
        pdb_path=pdb_path,
        chain_id="A",
        dist_cutoff=8.0,
        fully_connected=False,
    )

    if hasattr(rg, "build_residue_features_from_pdb"):
        x_full_np = rg.build_residue_features_from_pdb(seq, pdb_path, chain_id="A")
    else:
        x_full_np = rg.build_residue_features_from_sequence(seq)

    if x_full_np.size > 0:
        ss8_sasa_block = x_full_np[:, 24:33]                 # (L, 9)
        zero_mask = np.all(ss8_sasa_block == 0.0, axis=1)    # (L,)
        zero_ratio = float(zero_mask.mean())
        L = x_full_np.shape[0]

        print(
            f"  [SS8+SASA] protein={protein} L={L} "
            f"zero_residues={zero_mask.sum()} ({zero_ratio * 100:.1f}%)"
        )

        if zero_ratio > 0.5:
            print(
                f"  [WARN] protein {protein}: "
                f"{zero_ratio * 100:.1f}% residues have SS8+SASA all-zero (>50%)."
            )
            bad_proteins_stats.append(
                {
                    "protein": protein,
                    "L": int(L),
                    "num_zero_residues": int(zero_mask.sum()),
                    "zero_ratio": zero_ratio,
                }
            )

    x_full = torch.tensor(x_full_np, dtype=torch.float32)

    data = {
        "edge_index": gei,          # (2, E)
        "edge_attr": gea,           # (E, 2)
        "dist_matrix": dist_matrix, # (L, L) numpy
        "x_full": x_full,           # (L, 47) tensor
        "seq": seq,
    }
    out_path = os.path.join(OUT_DIR, f"{protein}.pt")
    torch.save(data, out_path)
    print(f"  [Save] {out_path}")

if bad_proteins_stats:
    bad_df = pd.DataFrame(bad_proteins_stats)
    bad_csv = os.path.join(OUT_DIR, "ss8_sasa_zero_over50.csv")
    bad_df.to_csv(bad_csv, index=False)
    print(f"\n[Summary] Proteins with SS8+SASA all-zero ratio > 0.5 saved to: {bad_csv}")
    print(bad_df)
else:
    print("\n[Summary] No protein has SS8+SASA all-zero ratio > 50%.")
