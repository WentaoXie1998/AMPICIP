import math
from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    TransformerConv,
    GINEConv,
    global_add_pool,
    global_max_pool,
)


class Attention(nn.Module):
    """
    Multi-head attention for 3D input [B, T, C] -> [B, T, C]
    """

    def __init__(self, embedding_dim: int, num_heads: int,
                 downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, c // self.num_heads)
        return x.transpose(1, 2)  # [B, H, T, C/H]

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, h, t, c_per_head = x.shape
        x = x.transpose(1, 2)  # [B, T, H, C/H]
        return x.reshape(b, t, h * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        q, k, v: [B, T, C]
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k)  # [B, H, T_q, T_k]
        attn = attn / math.sqrt(k.size(-1))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, H, T_q, C/H]
        out = self._recombine_heads(out)  # [B, T_q, C]
        out = self.out_proj(out)
        return out



class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_downsample_rate: int = 1,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        attn_out = self.attn(q, k, v)
        q = q + self.dropout(attn_out)
        q = self.norm(q)
        return q


class GraphTwoWayCrossAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        depth: int = 1,
        mlp_dim: Optional[int] = None,
        attention_downsample_rate: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        self.src_from_tgt = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_downsample_rate=attention_downsample_rate,
                    dropout_rate=dropout_rate,
                )
                for _ in range(depth)
            ]
        )
        self.tgt_from_src = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_downsample_rate=attention_downsample_rate,
                    dropout_rate=dropout_rate,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x_src: Tensor,
        batch_src: Tensor,
        x_tgt: Tensor,
        batch_tgt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if (
            batch_src is None
            or batch_tgt is None
            or batch_src.numel() == 0
            or batch_tgt.numel() == 0
        ):
            return x_src, x_tgt

        num_graphs = int(torch.max(torch.cat([batch_src, batch_tgt])).item()) + 1

        out_src = x_src
        out_tgt = x_tgt

        for layer_idx in range(self.depth):
            layer_src = self.src_from_tgt[layer_idx]
            layer_tgt = self.tgt_from_src[layer_idx]

            x_src_in = out_src
            x_tgt_in = out_tgt

            new_src = x_src_in.clone()
            new_tgt = x_tgt_in.clone()

            for g in range(num_graphs):
                idx_s = (batch_src == g).nonzero(as_tuple=False).view(-1)
                idx_t = (batch_tgt == g).nonzero(as_tuple=False).view(-1)
                if idx_s.numel() == 0 or idx_t.numel() == 0:
                    continue

                src_tokens_in = x_src_in[idx_s].unsqueeze(0)  # [1, T_src, C]
                tgt_tokens_in = x_tgt_in[idx_t].unsqueeze(0)  # [1, T_tgt, C]

                # 1) src ← tgt  (Q=src, K/V=tgt)
                src_tokens_out = layer_src(
                    q=src_tokens_in,
                    k=tgt_tokens_in,
                    v=tgt_tokens_in,
                )

                tgt_tokens_out = layer_tgt(
                    q=tgt_tokens_in,
                    k=src_tokens_in,
                    v=src_tokens_in,
                )

                new_src[idx_s] = src_tokens_out.squeeze(0)
                new_tgt[idx_t] = tgt_tokens_out.squeeze(0)

            out_src, out_tgt = new_src, new_tgt

        return out_src, out_tgt

class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            in_c = dims[i]
            out_c = dims[i + 1]
            conv = GATv2Conv(
                in_channels=in_c,
                out_channels=out_c,
                heads=heads,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(out_c))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        for conv, norm in zip(self.layers, self.norms):
            x_in = x  # residual

            if edge_index is not None and edge_index.numel() > 0:
                if edge_attr is not None and edge_attr.numel() > 0:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
                x = F.elu(x)
                x = self.dropout(x)

            if x.shape[-1] == x_in.shape[-1]:
                x = x + x_in

            x = norm(x)

        return x

class DualBranchRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_dim = edge_dim

        # 单分支 GAT
        self.gat = GATv2Conv(
            in_channels=dim,
            out_channels=dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        x_in = x

        if edge_index is not None and edge_index.numel() > 0:
            if edge_attr is not None and edge_attr.numel() > 0:
                x_msg = self.gat(x, edge_index, edge_attr)
            else:
                x_msg = self.gat(x, edge_index)
            x_msg = F.elu(x_msg)
            x_msg = self.dropout(x_msg)
        else:
            x_msg = x

        x_out = x_in + x_msg
        x_out = self.norm(x_out)
        return x_out



# ============================================================
# 6) Attention-based Graph Readout
# ============================================================
class AttnReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        if x.size(0) == 0:
            return x.new_zeros((0, x.size(-1)))

        scores = self.att(x).squeeze(-1)  # [N]

        # 数值稳定的 per-graph softmax
        max_scores = global_max_pool(scores, batch)         # [B]
        scores = scores - max_scores[batch]                 # [N]
        weights = scores.exp()
        denom = global_add_pool(weights, batch)[batch] + 1e-12
        weights = weights / denom                           # [N]

        out = global_add_pool(x * weights.unsqueeze(-1), batch)  # [B, C]
        return out


class TriViewFusion(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("fixed_alpha", torch.ones(1, 1))

    def forward(self, z_atom: Tensor) -> Tuple[Tensor, Tensor]:
        B = z_atom.size(0)
        z_drug = z_atom
        alpha = self.fixed_alpha.expand(B, -1)  # [B, 1]
        return z_drug, alpha



class DPIHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = embed_dim * 2  

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, z_drug: Tensor, z_protein: Tensor) -> Tuple[Tensor, Tensor]:
        feat = torch.cat([z_drug, z_protein], dim=-1)  # [B, 2C]
        logit = self.mlp(feat).squeeze(-1)             # [B]
        prob = torch.sigmoid(logit)
        return prob, logit


