"""
=============================================================================
model_tgnn.py  —  Temporal Graph Neural Network for GST Fraud Detection
=============================================================================
Paper: "Temporal Graph Neural Networks for Real-Time Detection of
         GST Circular Trading and Shell Entities"

Architecture Modules
--------------------
  1. BochnerTimeEncoder    – Learnable harmonic (Fourier) time encoding Φ(τ)
  2. DirectionalMessageMLP – MLP_src / MLP_dst for interaction message passing
  3. NodeMemory            – GRU-based dynamic memory s_i(t)
  4. TGATLayer             – Temporal multi-head neighbourhood self-attention
  5. TGNN                  – End-to-end model combining all above modules
                             with dual heads: node classifier + edge classifier
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bochner Harmonic Continuous-Time Encoder  Φ(τ)
# ─────────────────────────────────────────────────────────────────────────────
class BochnerTimeEncoder(nn.Module):
    """
    Encodes a scalar time delta τ into a d_t-dimensional Fourier feature vector.

    Φ(τ) = (1/√d_t) [cos(ω₁τ), sin(ω₁τ), …, cos(ω_{d_t/2}τ), sin(ω_{d_t/2}τ)]

    Parameters
    ----------
    d_t : int
        Output dimension (must be even).  Default = 64.
    """

    def __init__(self, d_t: int = 64):
        super().__init__()
        assert d_t % 2 == 0, "d_t must be even for cos/sin pairs."
        self.d_t = d_t
        # Learnable frequencies ω  ∈ ℝ^{d_t/2}
        self.omega = nn.Parameter(torch.randn(d_t // 2))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tau : Tensor of shape (...,)
            Scalar time deltas in seconds.

        Returns
        -------
        Tensor of shape (..., d_t)
        """
        # tau: (...,) → (..., 1) so we can broadcast with omega (d_t/2,)
        tau = tau.unsqueeze(-1)                          # (..., 1)
        angles = tau * self.omega                        # (..., d_t/2)
        phi = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., d_t)
        return phi / math.sqrt(self.d_t)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Directional Interaction Message MLPs
# ─────────────────────────────────────────────────────────────────────────────
class DirectionalMessageMLP(nn.Module):
    """
    Two separate MLPs (MLP_src, MLP_dst) that produce directional interaction
    messages for a transaction edge e_{ij}(t):

        m_i(t) = MLP_src( s_i ‖ s_j ‖ Φ(Δt_i) ‖ a_{ij} )
        m_j(t) = MLP_dst( s_j ‖ s_i ‖ Φ(Δt_j) ‖ a_{ij} )

    Parameters
    ----------
    mem_dim  : int  – dimension of node memory s_i
    time_dim : int  – dimension of time encoding Φ(τ)
    edge_dim : int  – dimension of edge feature vector a_{ij}
    msg_dim  : int  – output message dimension
    dropout  : float
    """

    def __init__(
        self,
        mem_dim: int,
        time_dim: int,
        edge_dim: int,
        msg_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = 2 * mem_dim + time_dim + edge_dim

        def _mlp():
            return nn.Sequential(
                nn.Linear(in_dim, msg_dim * 2),
                nn.LayerNorm(msg_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(msg_dim * 2, msg_dim),
                nn.GELU(),
            )

        self.mlp_src = _mlp()
        self.mlp_dst = _mlp()

    def forward(
        self,
        s_i: torch.Tensor,   # (E, mem_dim)  – source node memory
        s_j: torch.Tensor,   # (E, mem_dim)  – dest   node memory
        phi_i: torch.Tensor, # (E, time_dim) – Φ(t - t_i^last)
        phi_j: torch.Tensor, # (E, time_dim) – Φ(t - t_j^last)
        a_ij: torch.Tensor,  # (E, edge_dim) – edge features
    ):
        """Returns (m_i, m_j) each of shape (E, msg_dim)."""
        inp_src = torch.cat([s_i, s_j, phi_i, a_ij], dim=-1)
        inp_dst = torch.cat([s_j, s_i, phi_j, a_ij], dim=-1)
        return self.mlp_src(inp_src), self.mlp_dst(inp_dst)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Node Memory (GRU Cell)
# ─────────────────────────────────────────────────────────────────────────────
class NodeMemory(nn.Module):
    """
    Maintains per-node dynamic state vectors s_i(t) updated via a GRU:

        s_i(t) = GRU( m̄_i(t),  s_i(t⁻) )

    where m̄_i(t) is the aggregated incoming message.

    Parameters
    ----------
    n_nodes  : int  – total number of nodes in the graph
    mem_dim  : int  – dimension of each memory vector
    msg_dim  : int  – incoming message dimension (GRU input)
    """

    def __init__(self, n_nodes: int, mem_dim: int, msg_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.mem_dim = mem_dim
        # GRU cell: input = aggregated message, hidden = current state
        self.gru_cell = nn.GRUCell(input_size=msg_dim, hidden_size=mem_dim)
        # Raw memory buffer – NOT a parameter (maintained as a running state)
        self.register_buffer("memory", torch.zeros(n_nodes, mem_dim))
        self.register_buffer("last_update", torch.zeros(n_nodes))  # epoch secs

    def reset(self):
        """Zero all memories (call at episode start)."""
        self.memory.zero_()
        self.last_update.zero_()

    def get(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve current state for a batch of node indices."""
        return self.memory[node_ids]

    def update(self, node_ids: torch.Tensor, messages: torch.Tensor, t: torch.Tensor):
        """
        Parameters
        ----------
        node_ids : (B,)  – node indices to update
        messages : (B, msg_dim) – aggregated messages
        t        : (B,)  – current timestamps (epoch seconds)
        """
        # Get previous hidden state for these nodes
        h_prev = self.memory[node_ids]          # (B, mem_dim)
        h_new  = self.gru_cell(messages, h_prev) # (B, mem_dim)

        # Scatter update back to memory buffer (detached to prevent backprop through time)
        self.memory = self.memory.clone()
        self.memory[node_ids] = h_new.detach()
        self.last_update[node_ids] = t.detach()

    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return last-update timestamps for given nodes."""
        return self.last_update[node_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temporal Multi-Head Self-Attention (TGAT Layer)
# ─────────────────────────────────────────────────────────────────────────────
class TGATLayer(nn.Module):
    """
    Computes dynamic node embeddings z_i(t) by attending over temporal
    neighbours with time-aware key/query/value projections.

    For each node i with temporal neighbours  {(j, t_k)} ∈ N_i(t):

        q_i     = W_Q [ s_i(t) ‖ Φ(0) ]
        k_{ij}  = W_K [ s_j(t) ‖ Φ(t - t_k) ]
        v_{ij}  = W_V [ s_j(t) ‖ Φ(t - t_k) ‖ a_{ij} ]

    Multi-head attention → concatenate → linear projection → z_i(t)

    Parameters
    ----------
    mem_dim   : int  – node memory dimension
    time_dim  : int  – Bochner time encoding dimension
    edge_dim  : int  – edge feature dimension
    n_heads   : int  – number of attention heads
    out_dim   : int  – output embedding dimension
    dropout   : float
    """

    def __init__(
        self,
        mem_dim: int,
        time_dim: int,
        edge_dim: int,
        n_heads: int = 4,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads."
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        qk_in = mem_dim + time_dim   # query / key input dim
        v_in  = mem_dim + time_dim + edge_dim  # value input dim

        self.W_Q = nn.Linear(qk_in, out_dim, bias=False)
        self.W_K = nn.Linear(qk_in, out_dim, bias=False)
        self.W_V = nn.Linear(v_in,  out_dim, bias=False)
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        s_center: torch.Tensor,   # (N, mem_dim)   – center node memories
        s_neigh:  torch.Tensor,   # (N, K, mem_dim) – neighbour memories
        phi_zero: torch.Tensor,   # (N, time_dim)  – Φ(0) for center
        phi_delta: torch.Tensor,  # (N, K, time_dim) – Φ(t - t_k)
        a_neigh:  torch.Tensor,   # (N, K, edge_dim) – neighbour edge feats
        mask: torch.Tensor = None, # (N, K) bool mask – True = valid neighbour
    ) -> torch.Tensor:
        """
        Returns
        -------
        z : Tensor of shape (N, out_dim)
        """
        N, K, _ = s_neigh.shape

        # — Query (per center node) ————————————————————————
        q_in = torch.cat([s_center, phi_zero], dim=-1)      # (N, qk_in)
        Q = self.W_Q(q_in).view(N, 1, self.n_heads, self.head_dim)  # (N,1,H,d)
        Q = Q.transpose(1, 2)                               # (N, H, 1, d)

        # — Key (per neighbour) ————————————————————————————
        k_in = torch.cat([s_neigh, phi_delta], dim=-1)      # (N, K, qk_in)
        K_t  = self.W_K(k_in).view(N, K, self.n_heads, self.head_dim)
        K_t  = K_t.transpose(1, 2)                          # (N, H, K, d)

        # — Value (per neighbour) ——————————————————————————
        v_in = torch.cat([s_neigh, phi_delta, a_neigh], dim=-1) # (N, K, v_in)
        V    = self.W_V(v_in).view(N, K, self.n_heads, self.head_dim)
        V    = V.transpose(1, 2)                             # (N, H, K, d)

        # — Scaled dot-product attention ———————————————————
        attn = torch.matmul(Q, K_t.transpose(-2, -1)) / self.scale  # (N,H,1,K)

        if mask is not None:
            # mask: (N, K) → (N, 1, 1, K)
            invalid = ~mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(invalid, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN (all-invalid rows) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # — Aggregate ——————————————————————————————————————
        out = torch.matmul(attn, V)          # (N, H, 1, d)
        out = out.transpose(1, 2).reshape(N, -1)  # (N, out_dim)
        z = self.norm(self.dropout(self.proj(out)))
        return z


# ─────────────────────────────────────────────────────────────────────────────
# 5. End-to-End T-GNN Model
# ─────────────────────────────────────────────────────────────────────────────
class TGNN(nn.Module):
    """
    Temporal Graph Neural Network for GST Fraud Detection.

    Combines:
      • BochnerTimeEncoder  → time encoding Φ(τ)
      • DirectionalMessageMLP → interaction messages
      • NodeMemory (GRU)    → dynamic node state
      • TGATLayer           → temporal neighbourhood attention
      • Node head           → shell-entity binary classifier
      • Edge head           → fake-invoice binary classifier

    Parameters
    ----------
    n_nodes   : int  – total number of nodes in the graph
    node_feat_dim : int  – input static node feature dimension (d_node)
    edge_feat_dim : int  – edge feature dimension              (d_edge)
    mem_dim   : int  – node memory / GRU hidden dimension
    time_dim  : int  – Bochner time encoding dimension (must be even)
    msg_dim   : int  – interaction message dimension
    n_heads   : int  – number of attention heads in TGAT
    emb_dim   : int  – final embedding dimension
    dropout   : float
    n_layers  : int  – number of stacked TGAT layers
    """

    def __init__(
        self,
        n_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        mem_dim: int    = 64,
        time_dim: int   = 64,
        msg_dim: int    = 64,
        n_heads: int    = 4,
        emb_dim: int    = 64,
        dropout: float  = 0.2,
        n_layers: int   = 2,
    ):
        super().__init__()
        self.mem_dim   = mem_dim
        self.time_dim  = time_dim
        self.msg_dim   = msg_dim
        self.emb_dim   = emb_dim
        self.n_layers  = n_layers

        # — Static node feature projection ————————————————
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, mem_dim),
            nn.LayerNorm(mem_dim),
            nn.GELU(),
        )

        # — Time encoder ——————————————————————————————————
        self.time_enc = BochnerTimeEncoder(time_dim)

        # — Node memory module ————————————————————————————
        self.memory = NodeMemory(n_nodes, mem_dim, msg_dim)

        # — Directional message MLPs ———————————————————————
        self.msg_mlp = DirectionalMessageMLP(
            mem_dim=mem_dim,
            time_dim=time_dim,
            edge_dim=edge_feat_dim,
            msg_dim=msg_dim,
            dropout=dropout,
        )

        # — Stacked TGAT layers ————————————————————————————
        self.tgat_layers = nn.ModuleList([
            TGATLayer(
                mem_dim=mem_dim,
                time_dim=time_dim,
                edge_dim=edge_feat_dim,
                n_heads=n_heads,
                out_dim=emb_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Layer-to-layer projection (align dims for residual)
        self.layer_proj = nn.ModuleList([
            nn.Linear(emb_dim, mem_dim) for _ in range(n_layers - 1)
        ])

        # — Node classification head ——————————————————————
        self.node_head = nn.Sequential(
            nn.Linear(emb_dim + mem_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # — Edge classification head ——————————————————————
        self.edge_head = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    # ── Utility helpers ────────────────────────────────────────────────────

    @staticmethod
    def _scatter_mean(src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
        """Mean-aggregation: aggregate src rows by index into n slots."""
        out   = torch.zeros(n, src.size(1), device=src.device)
        count = torch.zeros(n, 1, device=src.device)
        idx_e = idx.unsqueeze(-1).expand_as(src)
        out.scatter_add_(0, idx_e, src)
        count.scatter_add_(0, idx.unsqueeze(-1), torch.ones(len(idx), 1, device=src.device))
        count = count.clamp(min=1)
        return out / count

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(
        self,
        x_node: torch.Tensor,          # (N, d_node)  – static node features
        edge_index: torch.Tensor,       # (2, E)
        edge_attr: torch.Tensor,        # (E, d_edge)
        timestamps: torch.Tensor,       # (E,)          – edge timestamps (seconds)
        neighbor_index: torch.Tensor,   # (N, K)        – K temporal neighbours per node
        neighbor_attr: torch.Tensor,    # (N, K, d_edge)
        neighbor_ts: torch.Tensor,      # (N, K)        – timestamps of those edges
        t_query: float,                 # scalar query time
        valid_mask: torch.Tensor,       # (N, K) bool   – True = valid slot
    ):
        """
        Parameters
        ----------
        All tensors live on the same device.

        Returns
        -------
        logit_node : (N,)    – raw logits for shell-entity classification
        logit_edge : (E,)    – raw logits for fake-invoice classification
        z_node     : (N, emb_dim) – final node embeddings
        """
        device = x_node.device
        N = x_node.size(0)
        E = edge_index.size(1)
        t_q = torch.tensor(t_query, device=device, dtype=torch.float)

        # ① Project static node features into memory space
        x_proj = self.node_proj(x_node)    # (N, mem_dim)

        # ② Retrieve current dynamic memory + blend with static features
        all_ids = torch.arange(N, device=device)
        s_mem   = self.memory.get(all_ids)  # (N, mem_dim)
        s       = s_mem + x_proj            # (N, mem_dim)   — residual blend

        # ③ Process each edge's interaction to update source/dest memories
        if E > 0:
            src_ids = edge_index[0]   # (E,)
            dst_ids = edge_index[1]   # (E,)
            s_src   = s[src_ids]      # (E, mem_dim)
            s_dst   = s[dst_ids]      # (E, mem_dim)

            # Time deltas for source and destination nodes
            t_src_last = self.memory.get_last_update(src_ids)  # (E,)
            t_dst_last = self.memory.get_last_update(dst_ids)  # (E,)
            dt_src = (timestamps - t_src_last).clamp(min=0.0)  # (E,)
            dt_dst = (timestamps - t_dst_last).clamp(min=0.0)  # (E,)

            phi_src = self.time_enc(dt_src)   # (E, time_dim)
            phi_dst = self.time_enc(dt_dst)   # (E, time_dim)

            m_src, m_dst = self.msg_mlp(s_src, s_dst, phi_src, phi_dst, edge_attr)
            # m_src: (E, msg_dim),  m_dst: (E, msg_dim)

            # Aggregate messages per node
            agg_src = self._scatter_mean(m_src, src_ids, N)  # (N, msg_dim)
            agg_dst = self._scatter_mean(m_dst, dst_ids, N)  # (N, msg_dim)
            agg     = agg_src + agg_dst                       # simple sum

            # Update memory (detached – memory is a side-effect)
            # Process unique node ids that actually received messages
            active = torch.unique(torch.cat([src_ids, dst_ids]))
            t_mean = timestamps.float()
            t_scalar = t_mean.mean() if E > 0 else t_q
            self.memory.update(
                active,
                agg[active],
                t_scalar.expand(len(active)),
            )
            # Refresh s with updated memory
            s_mem = self.memory.get(all_ids)   # (N, mem_dim)
            s     = s_mem + x_proj             # (N, mem_dim)

        # ④ TGAT: compute z_i(t) using temporal neighbourhood attention
        z = s  # start with blended memory
        for layer_idx, tgat in enumerate(self.tgat_layers):
            K = neighbor_index.size(1)

            # Gather neighbour memories
            neigh_ids_flat = neighbor_index.reshape(-1)            # (N*K,)
            neigh_ids_flat = neigh_ids_flat.clamp(0, N - 1)        # safety clamp
            s_neigh_flat   = s[neigh_ids_flat]                     # (N*K, mem_dim)
            s_neigh        = s_neigh_flat.view(N, K, self.mem_dim) # (N, K, mem_dim)

            # Time encodings
            phi_zero  = self.time_enc(torch.zeros(N, device=device))     # (N, time_dim)
            dt_neigh  = (t_q - neighbor_ts).clamp(min=0.0)              # (N, K)
            phi_delta = self.time_enc(dt_neigh.reshape(-1)).view(N, K, self.time_dim)

            z_new = tgat(z if layer_idx == 0 else z_proj,
                         s_neigh, phi_zero, phi_delta,
                         neighbor_attr, valid_mask)              # (N, emb_dim)
            if layer_idx < self.n_layers - 1:
                z_proj = self.layer_proj[layer_idx](z_new)      # (N, mem_dim)
            z = z_new                                            # final is emb_dim

        # z: (N, emb_dim)

        # ⑤ Node classification head
        node_in     = torch.cat([z, s_mem], dim=-1)   # (N, emb_dim + mem_dim)
        logit_node  = self.node_head(node_in).squeeze(-1)   # (N,)

        # ⑥ Edge classification head
        if E > 0:
            z_src_e   = z[edge_index[0]]               # (E, emb_dim)
            z_dst_e   = z[edge_index[1]]               # (E, emb_dim)
            edge_in   = torch.cat([z_src_e, z_dst_e, edge_attr], dim=-1)
            logit_edge = self.edge_head(edge_in).squeeze(-1)    # (E,)
        else:
            logit_edge = torch.zeros(0, device=device)

        return logit_node, logit_edge, z


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced edge-level classification.
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        bce    = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl     = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return fl.mean() if self.reduction == "mean" else fl.sum()


class CycleConservationLoss(nn.Module):
    """
    Penalises closed circular loops where:
      - Value addition is near zero  (|Σ_in - Σ_out| ≈ 0)
      - Execution time is short      (cycle_duration ≤ threshold in seconds)

    Loss = E_C [ sigmoid(logit_avg) · exp(-|val_net| / scale) · (t_last - t_first ≤ τ_max) ]
    """

    def __init__(self, tau_max_hours: float = 6.0, val_scale: float = 1e5):
        super().__init__()
        self.tau_max   = tau_max_hours * 3600.0   # convert to seconds
        self.val_scale = val_scale

    def forward(
        self,
        cycle_edge_logits: torch.Tensor,   # (E_cycle,)   – raw logits for cycle edges
        cycle_amounts:     torch.Tensor,   # (E_cycle,)   – taxable amounts
        cycle_ts:          torch.Tensor,   # (E_cycle,)   – timestamps (seconds)
        cycle_ids:         torch.Tensor,   # (E_cycle,)   – integer cycle membership id
    ) -> torch.Tensor:
        if cycle_edge_logits.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=cycle_edge_logits.device
                                if cycle_edge_logits.numel() > 0 else torch.device("cpu"))

        unique_cycles = cycle_ids.unique()
        losses = []
        for cid in unique_cycles:
            mask       = cycle_ids == cid
            logits_c   = cycle_edge_logits[mask]    # (k,)
            amounts_c  = cycle_amounts[mask]         # (k,)
            ts_c       = cycle_ts[mask]              # (k,)

            # Mean fraud probability for the cycle
            p_fraud = torch.sigmoid(logits_c).mean()

            # Value-addition conservation (near-zero = suspicious)
            val_in  = amounts_c.sum()
            val_net = amounts_c[1:] - amounts_c[:-1]
            val_pen = torch.exp(-val_net.abs().mean() / self.val_scale)

            # Fast execution penalty
            duration   = ts_c.max() - ts_c.min() + 1e-6
            speed_pen  = (duration <= self.tau_max).float()

            cycle_loss = p_fraud * val_pen * speed_pen
            losses.append(cycle_loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)


class MultiTaskLoss(nn.Module):
    """
    L_total = λ1·L_node + λ2·L_edge + λ3·L_cycle + λ4·‖Θ‖²

    Default λ weighting follows paper recommendations.
    """

    def __init__(
        self,
        pos_weight_node: float = 12.0,  # ≈ 1000/80 imbalance ratio
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 0.5,
        lambda4: float = 1e-4,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

        self.focal      = FocalLoss(alpha=0.25, gamma=2.0)
        self.cycle_loss = CycleConservationLoss()
        # pos_weight as a buffer so it moves with .to(device)
        self.register_buffer("pos_weight", torch.tensor([pos_weight_node]))

    def forward(
        self,
        logit_node:   torch.Tensor,   # (N,)
        y_node:       torch.Tensor,   # (N,) int
        logit_edge:   torch.Tensor,   # (E,)
        y_edge:       torch.Tensor,   # (E,) int
        model:        nn.Module,
        # Optional cycle-specific tensors
        cycle_logits: torch.Tensor  = None,
        cycle_amounts: torch.Tensor = None,
        cycle_ts:     torch.Tensor  = None,
        cycle_ids:    torch.Tensor  = None,
    ) -> dict:
        # ── Node loss (Weighted BCE) ───────────────────────
        l_node = F.binary_cross_entropy_with_logits(
            logit_node, y_node.float(), pos_weight=self.pos_weight
        )

        # ── Edge loss (Focal) ──────────────────────────────
        l_edge = self.focal(logit_edge, y_edge) if logit_edge.numel() > 0 \
                 else torch.tensor(0.0, device=logit_node.device)

        # ── Cycle conservation loss ────────────────────────
        if cycle_logits is not None and cycle_logits.numel() > 0:
            l_cycle = self.cycle_loss(cycle_logits, cycle_amounts, cycle_ts, cycle_ids)
        else:
            l_cycle = torch.tensor(0.0, device=logit_node.device)

        # ── L2 regularisation ──────────────────────────────
        l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)

        total = (
            self.lambda1 * l_node
            + self.lambda2 * l_edge
            + self.lambda3 * l_cycle
            + self.lambda4 * l2
        )

        return {
            "total": total,
            "node":  l_node,
            "edge":  l_edge,
            "cycle": l_cycle,
            "l2":    l2,
        }
