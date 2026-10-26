from typing import Dict, Any, Optional

import torch
from torch import nn, Tensor

from layers import (
    GATEncoder,
    GraphTwoWayCrossAttention,
    DualBranchRefiner,
    TriViewFusion,
    DPIHead,
    AttnReadout,
)

HIDDEN_DIM = 256

ATOM_IN_DIM = 55
RES_IN_DIM = 47

ATOM_EDGE_DIM = 16
RES_EDGE_DIM = 2


class ProteinESMFusion(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_res: Tensor, z_esm: Tensor) -> (Tensor, Tensor):
        h = torch.cat([z_res, z_esm], dim=-1)   # [B, 2C]
        gate = self.sigmoid(self.gate_mlp(h))   # [B, C] in (0,1)

        z_esm_gated = gate * z_esm
        z_protein = z_res + z_esm_gated

        return z_protein, gate


class DPIMultiViewModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        gat_heads: int = 4,
        dropout: float = 0.1,
        atom_in_dim: int = ATOM_IN_DIM,
        res_in_dim: int = RES_IN_DIM,
        atom_edge_dim: int = ATOM_EDGE_DIM,
        res_edge_dim: int = RES_EDGE_DIM,
        protein_esm_dim: int = 1280,
        use_protein_esm: bool = True,
        cip_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.gat_heads = int(gat_heads)
        self.use_protein_esm = bool(use_protein_esm)
        self.protein_esm_dim = int(protein_esm_dim)
        self.cip_layers = int(cip_layers)

        if self.cip_layers < 1:
            raise ValueError(f"cip_layers must be >= 1, got {self.cip_layers}")

        self.atom_encoder = GATEncoder(
            in_dim=atom_in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=2,
            heads=self.gat_heads,
            edge_dim=atom_edge_dim,
            dropout=dropout,
        )
        self.res_encoder = GATEncoder(
            in_dim=res_in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=2,
            heads=self.gat_heads,
            edge_dim=res_edge_dim,
            dropout=dropout,
        )

        self.cross_atom_res = GraphTwoWayCrossAttention(
            embedding_dim=self.hidden_dim,
            num_heads=self.gat_heads,
            depth=1,
            mlp_dim=self.hidden_dim * 4,
            dropout_rate=dropout,
        )

        self.atom_refiners = nn.ModuleList([
            DualBranchRefiner(
                dim=self.hidden_dim,
                heads=self.gat_heads,
                edge_dim=atom_edge_dim,
                dropout=dropout,
            )
            for _ in range(self.cip_layers)
        ])

        self.res_refiners = nn.ModuleList([
            DualBranchRefiner(
                dim=self.hidden_dim,
                heads=self.gat_heads,
                edge_dim=res_edge_dim,
                dropout=dropout,
            )
            for _ in range(self.cip_layers)
        ])

        self.atom_readout = AttnReadout(self.hidden_dim)
        self.res_readout = AttnReadout(self.hidden_dim)

        if self.use_protein_esm:
            mid_dim = self.hidden_dim * 4

            self.protein_esm_proj = nn.Sequential(
                nn.LayerNorm(self.protein_esm_dim),
                nn.Linear(self.protein_esm_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(mid_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.protein_fusion = ProteinESMFusion(
                embed_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim * 2,
            )
        else:
            self.protein_esm_proj = None
            self.protein_fusion = None

        self.fusion = TriViewFusion(
            embed_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
        )

        self.head = DPIHead(
            embed_dim=self.hidden_dim,
            hidden_dim1=self.hidden_dim * 2,
            hidden_dim2=self.hidden_dim // 2,
            dropout=0.2,
        )

    def forward(
        self,
        atom_data,
        res_data,
        protein_esm: Optional[Tensor] = None,
    ) -> Dict[str, Any]:

        xa = self.atom_encoder(
            atom_data.x,
            atom_data.edge_index,
            getattr(atom_data, "edge_attr", None),
        )
        xr = self.res_encoder(
            res_data.x,
            res_data.edge_index,
            getattr(res_data, "edge_attr", None),
        )

        atom_batch = atom_data.batch
        res_batch = res_data.batch

        xa, xr = self.cross_atom_res(
            x_src=xa,
            batch_src=atom_batch,
            x_tgt=xr,
            batch_tgt=res_batch,
        )

        for atom_refiner, res_refiner in zip(self.atom_refiners, self.res_refiners):
            xa = atom_refiner(
                xa,
                atom_data.edge_index,
                getattr(atom_data, "edge_attr", None),
            )
            xr = res_refiner(
                xr,
                res_data.edge_index,
                getattr(res_data, "edge_attr", None),
            )

        z_atom = self.atom_readout(xa, atom_batch)     # [B, C]
        z_res = self.res_readout(xr, res_batch)        # [B, C]

        z_esm_proj = None
        protein_gate = None

        if self.use_protein_esm and (protein_esm is not None):
            z_esm_proj = self.protein_esm_proj(protein_esm)      # [B, C]
            z_protein, protein_gate = self.protein_fusion(z_res, z_esm_proj)
        else:
            z_protein = z_res

        z_drug, gate_alpha = self.fusion(z_atom)

        prob, logit = self.head(z_drug, z_protein)

        return {
            "prob": prob,
            "logit": logit,
            "z_drug": z_drug,
            "z_atom": z_atom,
            "z_res": z_res,
            "z_protein": z_protein,
            "z_esm_proj": z_esm_proj,
            "gate_alpha": gate_alpha,
            "protein_gate": protein_gate,
        }
