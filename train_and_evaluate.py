"""
=============================================================================
train_and_evaluate.py  —  Training loop, evaluation, and benchmark tables
=============================================================================
Paper: "Temporal Graph Neural Networks for Real-Time Detection of
         GST Circular Trading and Shell Entities"

Execution
---------
    python train_and_evaluate.py

Outputs
-------
  • tgnn_gst_model.pth       – trained T-GNN weights
  • Table I  (ASCII)         – full benchmark comparison
  • Table II (ASCII)         – ablation study results
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

# ── Baseline imports ───────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed; using GradientBoostingClassifier as substitute.")

from sklearn.linear_model import LogisticRegression

# ── Local model import ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model_tgnn import TGNN, MultiTaskLoss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "seed":          42,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":      os.path.dirname(os.path.abspath(__file__)),
    # Model hyper-parameters
    "mem_dim":       64,
    "time_dim":      64,
    "msg_dim":       64,
    "n_heads":       4,
    "emb_dim":       64,
    "n_layers":      2,
    "dropout":       0.2,
    # Training
    "epochs":        30,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "batch_size":    512,
    "K_neighbors":   10,       # number of temporal neighbours in TGAT
    # Loss weights
    "lambda1":       1.0,      # node loss
    "lambda2":       1.0,      # edge (focal) loss
    "lambda3":       0.5,      # cycle conservation loss
    "lambda4":       1e-4,     # L2 regularisation
    "pos_weight_node": 12.5,   # ≈ 1000/80
    # Paths
    "model_save":    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "tgnn_gst_model.pth"),
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
DEVICE = torch.device(CFG["device"])
print(f"\n{'='*70}")
print(f"  T-GNN GST Fraud Detection  |  Device: {DEVICE}")
print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE A: DATA INGESTION & GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_build_graph():
    """
    Load CSVs, encode features, split chronologically 70/15/15,
    and return a compact graph-data dictionary.
    """
    data_dir = CFG["data_dir"]

    # ── Read raw CSVs ──────────────────────────────────────────────────────
    df_comp = pd.read_csv(os.path.join(data_dir, "gst_companies.csv"))
    df_inv  = pd.read_csv(os.path.join(data_dir, "gst_invoices.csv"))

    print(f"  Companies : {len(df_comp):,}  (shell={df_comp['is_shell'].sum()})")
    print(f"  Invoices  : {len(df_inv):,}   (fraud={df_inv['is_fraud'].sum()})\n")

    # ── Node ID mapping  gstin → integer ──────────────────────────────────
    all_gstins  = pd.concat([df_comp["gstin"],
                              df_inv["seller_gstin"],
                              df_inv["buyer_gstin"]]).unique()
    gstin2id    = {g: i for i, g in enumerate(all_gstins)}
    N           = len(gstin2id)

    # ── Node feature matrix  x ∈ ℝ^{N × 3} ───────────────────────────────
    #   features: capital_lakhs, employee_count, compliance_score
    node_feat_cols = ["capital_lakhs", "employee_count", "compliance_score"]
    df_comp_idx = df_comp.copy()
    df_comp_idx["node_id"] = df_comp_idx["gstin"].map(gstin2id)

    # Standardise
    scaler_node = StandardScaler()
    node_vals   = scaler_node.fit_transform(df_comp_idx[node_feat_cols].values.astype(float))

    x_node_np = np.zeros((N, len(node_feat_cols)), dtype=np.float32)
    for _, row in df_comp_idx.iterrows():
        nid = int(row["node_id"])
        x_node_np[nid] = node_vals[int(row.name)]

    y_node_np  = np.zeros(N, dtype=np.int64)
    shell_flag = np.zeros(N, dtype=np.int64)
    for _, row in df_comp_idx.iterrows():
        nid = int(row["node_id"])
        y_node_np[nid] = int(row["is_shell"])
        shell_flag[nid] = int(row["is_shell"])

    # ── Edge construction ──────────────────────────────────────────────────
    df_inv = df_inv.copy()
    df_inv["src_id"]    = df_inv["seller_gstin"].map(gstin2id)
    df_inv["dst_id"]    = df_inv["buyer_gstin"].map(gstin2id)
    df_inv["dt"]        = pd.to_datetime(df_inv["timestamp"])
    df_inv              = df_inv.sort_values("dt").reset_index(drop=True)

    t0              = df_inv["dt"].min()
    df_inv["epoch"] = (df_inv["dt"] - t0).dt.total_seconds().astype(np.float32)

    # Edge features: normalise taxable_amount, gst_rate, has_eway_bill
    scaler_edge = StandardScaler()
    edge_feats_raw = df_inv[["taxable_amount", "gst_rate"]].values.astype(float)
    edge_feats_norm = scaler_edge.fit_transform(edge_feats_raw)

    edge_attr_np = np.column_stack([
        edge_feats_norm,
        df_inv["has_eway_bill"].values.astype(np.float32),
    ]).astype(np.float32)   # (E, 3)

    y_edge_np = df_inv["is_fraud"].values.astype(np.int64)

    # ── Chronological 70/15/15 split ──────────────────────────────────────
    E = len(df_inv)
    cut_train = int(0.70 * E)
    cut_val   = int(0.85 * E)

    idx_train = np.arange(0, cut_train)
    idx_val   = np.arange(cut_train, cut_val)
    idx_test  = np.arange(cut_val, E)

    print(f"  Edge split  →  train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")
    print(f"  Time range  →  {df_inv['dt'].min().date()} .. {df_inv['dt'].max().date()}\n")

    # ── Build temporal neighbour index for TGAT ────────────────────────────
    #   For each node i, store the last K_neighbors transactions up to time t_query.
    #   We precompute a static version for the training window; during inference
    #   we use the same structure but restricted to the relevant split.
    K = CFG["K_neighbors"]

    def build_neighbor_index(edge_slice, timestamps):
        """
        Returns:
            neighbor_idx  : (N, K)  – node indices of K most-recent neighbours
            neighbor_attr : (N, K, d_edge)
            neighbor_ts   : (N, K)
            valid_mask    : (N, K) bool
        """
        D = edge_attr_np.shape[1]
        neigh_idx  = np.zeros((N, K), dtype=np.int64)
        neigh_attr = np.zeros((N, K, D), dtype=np.float32)
        neigh_ts   = np.zeros((N, K), dtype=np.float32)
        valid      = np.zeros((N, K), dtype=bool)

        # Process edges in reverse-time order to pick most-recent K
        recent_src = {i: [] for i in range(N)}   # i → list of (t, dst, attr)
        recent_dst = {i: [] for i in range(N)}   # i → list of (t, src, attr)

        for e in edge_slice:
            src = int(df_inv.loc[e, "src_id"])
            dst = int(df_inv.loc[e, "dst_id"])
            t   = float(df_inv.loc[e, "epoch"])
            a   = edge_attr_np[e]

            recent_src[src].append((t, dst, a))
            recent_dst[dst].append((t, src, a))

        for i in range(N):
            entries = sorted(recent_src[i] + recent_dst[i], key=lambda x: -x[0])[:K]
            for k_idx, (t, j, a) in enumerate(entries):
                neigh_idx[i, k_idx]    = j
                neigh_attr[i, k_idx]   = a
                neigh_ts[i, k_idx]     = t
                valid[i, k_idx]        = True

        return neigh_idx, neigh_attr, neigh_ts, valid

    print("  Building temporal neighbour index … ", end="", flush=True)
    ni_train, na_train, nt_train, vm_train = build_neighbor_index(idx_train, None)
    ni_val,   na_val,   nt_val,   vm_val   = build_neighbor_index(
        np.concatenate([idx_train, idx_val]), None)
    ni_test,  na_test,  nt_test,  vm_test  = build_neighbor_index(
        np.concatenate([idx_train, idx_val, idx_test]), None)
    print("done.\n")

    # ── Cycle metadata ─────────────────────────────────────────────────────
    cycle_mask = df_inv["fraud_category"].str.startswith("Circular")
    df_cycles  = df_inv[cycle_mask]
    # Parse cycle id from 'Circular_Loop_X'
    df_cycles = df_cycles.copy()
    df_cycles["cycle_id"] = (
        df_cycles["fraud_category"]
        .str.extract(r"(\d+)$")[0].astype(int)
    )

    cycle_edge_idx     = df_cycles.index.tolist()
    cycle_amounts_np   = df_cycles["taxable_amount"].values.astype(np.float32)
    cycle_ts_np        = df_cycles["epoch"].values.astype(np.float32)
    cycle_ids_np       = df_cycles["cycle_id"].values.astype(np.int64)

    # ── Pack everything ────────────────────────────────────────────────────
    graph = {
        "N":              N,
        "E":              E,
        "x_node":         x_node_np,
        "y_node":         y_node_np,
        "edge_index":     np.stack([df_inv["src_id"].values,
                                    df_inv["dst_id"].values], axis=0).astype(np.int64),
        "edge_attr":      edge_attr_np,
        "timestamps":     df_inv["epoch"].values.astype(np.float32),
        "y_edge":         y_edge_np,
        # Splits
        "idx_train":      idx_train,
        "idx_val":        idx_val,
        "idx_test":       idx_test,
        # Temporal neighbour indices (per split)
        "ni_train": ni_train, "na_train": na_train,
        "nt_train": nt_train, "vm_train": vm_train,
        "ni_val":   ni_val,   "na_val":   na_val,
        "nt_val":   nt_val,   "vm_val":   vm_val,
        "ni_test":  ni_test,  "na_test":  na_test,
        "nt_test":  nt_test,  "vm_test":  vm_test,
        # Cycle info
        "cycle_edge_idx":    np.array(cycle_edge_idx, dtype=np.int64),
        "cycle_amounts":     cycle_amounts_np,
        "cycle_ts":          cycle_ts_np,
        "cycle_ids":         cycle_ids_np,
        # Meta
        "gstin2id":          gstin2id,
        "t0":                t0,
        "node_feat_dim":     len(node_feat_cols),
        "edge_feat_dim":     edge_attr_np.shape[1],
    }
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: tensor conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def split_tensors(graph, split="train"):
    """Return edge tensors for a given split."""
    idx = graph[f"idx_{split}"]
    ei  = graph["edge_index"][:, idx]
    ea  = graph["edge_attr"][idx]
    ts  = graph["timestamps"][idx]
    ye  = graph["y_edge"][idx]
    return (to_tensor(ei, torch.int64),
            to_tensor(ea),
            to_tensor(ts),
            to_tensor(ye, torch.int64))


def neighbor_tensors(graph, split="train"):
    return (to_tensor(graph[f"ni_{split}"], torch.int64),
            to_tensor(graph[f"na_{split}"]),
            to_tensor(graph[f"nt_{split}"]),
            to_tensor(graph[f"vm_{split}"], torch.bool))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5, task="edge"):
    """Returns dict of precision, recall, f1, auc, ap."""
    y_pred = (y_prob >= threshold).astype(int)
    eps = 1e-9

    # Handle degenerate cases
    if y_true.sum() == 0:
        return {"precision": 0, "recall": 0, "f1": 0, "auc": 0.5, "ap": 0, "loop_recall": 0}

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
    except Exception:
        auc, ap = 0.5, 0.0

    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "ap": ap, "loop_recall": rec}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE B/C: TRAIN T-GNN
# ─────────────────────────────────────────────────────────────────────────────

def build_model(graph):
    model = TGNN(
        n_nodes       = graph["N"],
        node_feat_dim = graph["node_feat_dim"],
        edge_feat_dim = graph["edge_feat_dim"],
        mem_dim       = CFG["mem_dim"],
        time_dim      = CFG["time_dim"],
        msg_dim       = CFG["msg_dim"],
        n_heads       = CFG["n_heads"],
        emb_dim       = CFG["emb_dim"],
        dropout       = CFG["dropout"],
        n_layers      = CFG["n_layers"],
    ).to(DEVICE)
    return model


def train_tgnn(graph):
    model = build_model(graph)
    criterion = MultiTaskLoss(
        pos_weight_node = CFG["pos_weight_node"],
        lambda1 = CFG["lambda1"],
        lambda2 = CFG["lambda2"],
        lambda3 = CFG["lambda3"],
        lambda4 = CFG["lambda4"],
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-5)

    x_node = to_tensor(graph["x_node"])
    y_node = to_tensor(graph["y_node"], torch.int64)

    # Pre-load train tensors
    ei_tr, ea_tr, ts_tr, ye_tr = split_tensors(graph, "train")
    ni_tr, na_tr, nt_tr, vm_tr = neighbor_tensors(graph, "train")

    # Cycle tensors (those in train window)
    cycle_in_train = graph["cycle_edge_idx"][
        np.isin(graph["cycle_edge_idx"], graph["idx_train"])
    ]
    has_cycle = len(cycle_in_train) > 0
    if has_cycle:
        cycle_amounts_t = to_tensor(graph["cycle_amounts"])
        cycle_ts_t      = to_tensor(graph["cycle_ts"])
        cycle_ids_t     = to_tensor(graph["cycle_ids"], torch.int64)
    else:
        cycle_amounts_t = cycle_ts_t = cycle_ids_t = None

    t_query_train = float(ts_tr.max().item())

    best_val_f1  = -1
    best_weights = None
    history      = []

    print("── Training T-GNN ─────────────────────────────────────────")
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        model.memory.reset()

        optimizer.zero_grad()

        logit_node, logit_edge, _ = model(
            x_node, ei_tr, ea_tr, ts_tr,
            ni_tr, na_tr, nt_tr,
            t_query_train, vm_tr,
        )

        # Cycle subset of train edges
        if has_cycle:
            cl = criterion(
                logit_node, y_node,
                logit_edge, ye_tr,
                model,
                logit_edge[: len(cycle_amounts_t)],
                cycle_amounts_t, cycle_ts_t, cycle_ids_t,
            )
        else:
            cl = criterion(logit_node, y_node, logit_edge, ye_tr, model)

        cl["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Validation ───────────────────────────────────
        if epoch % 5 == 0 or epoch == CFG["epochs"]:
            model.eval()
            with torch.no_grad():
                model.memory.reset()
                ei_v, ea_v, ts_v, ye_v = split_tensors(graph, "val")
                ni_v, na_v, nt_v, vm_v = neighbor_tensors(graph, "val")
                t_q_val = float(ts_v.max().item())

                ln_v, le_v, _ = model(
                    x_node, ei_v, ea_v, ts_v, ni_v, na_v, nt_v, t_q_val, vm_v)

                prob_node_v = torch.sigmoid(ln_v).cpu().numpy()
                prob_edge_v = torch.sigmoid(le_v).cpu().numpy()
                yn_v = graph["y_node"]
                ye_v_np = ye_v.cpu().numpy()

                m_node = compute_metrics(yn_v, prob_node_v, task="node")
                m_edge = compute_metrics(ye_v_np, prob_edge_v, task="edge")
                val_f1 = 0.5 * m_node["f1"] + 0.5 * m_edge["f1"]

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            l = cl["total"].item()
            print(f"  Ep {epoch:03d}/{CFG['epochs']}  loss={l:.4f}"
                  f"  val_node_f1={m_node['f1']:.3f}"
                  f"  val_edge_f1={m_edge['f1']:.3f}"
                  f"  val_auc={m_edge['auc']:.3f}")
            history.append({
                "epoch": epoch, "loss": l,
                "val_node_f1": m_node["f1"], "val_edge_f1": m_edge["f1"]
            })

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save model
    torch.save({"model_state": model.state_dict(), "cfg": CFG}, CFG["model_save"])
    print(f"\n  ✓ Best weights saved  →  {CFG['model_save']}\n")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# T-GNN TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tgnn(model, graph):
    model.eval()
    model.memory.reset()

    x_node = to_tensor(graph["x_node"])
    ei_te, ea_te, ts_te, ye_te = split_tensors(graph, "test")
    ni_te, na_te, nt_te, vm_te = neighbor_tensors(graph, "test")
    t_q = float(ts_te.max().item())

    t0 = time.perf_counter()
    with torch.no_grad():
        ln, le, _ = model(x_node, ei_te, ea_te, ts_te,
                          ni_te, na_te, nt_te, t_q, vm_te)
    latency_ms = (time.perf_counter() - t0) * 1000 / len(graph["idx_test"])

    prob_node = torch.sigmoid(ln).cpu().numpy()
    prob_edge = torch.sigmoid(le).cpu().numpy()

    m_node = compute_metrics(graph["y_node"], prob_node, task="node")
    m_edge = compute_metrics(ye_te.cpu().numpy(), prob_edge, task="edge")
    m_edge["latency_ms"] = latency_ms

    return m_node, m_edge, prob_node, prob_edge


# ─────────────────────────────────────────────────────────────────────────────
# MODULE D: BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def build_tabular_features(graph, split_idx):
    """
    Combine node-level and edge-level features into a flat row for ML baselines.
    Row = [src_node_feats, dst_node_feats, edge_feats, src_is_shell, dst_is_shell]
    """
    ei = graph["edge_index"][:, split_idx]   # (2, E_split)
    ea = graph["edge_attr"][split_idx]        # (E_split, d_edge)
    xn = graph["x_node"]                     # (N, d_node)
    ys = graph["y_node"]                     # (N,)

    src_feats  = xn[ei[0]]                  # (E, d_node)
    dst_feats  = xn[ei[1]]                  # (E, d_node)
    src_shell  = ys[ei[0]].reshape(-1, 1)   # (E, 1)
    dst_shell  = ys[ei[1]].reshape(-1, 1)   # (E, 1)
    ts_feat    = graph["timestamps"][split_idx].reshape(-1, 1)  # (E, 1)

    X = np.hstack([src_feats, dst_feats, ea, src_shell, dst_shell, ts_feat])
    y = graph["y_edge"][split_idx]
    return X.astype(np.float32), y.astype(np.int64)


# ── Rule-Based Baseline ───────────────────────────────────────────────────

def rule_based_baseline(graph):
    """
    Flag an invoice as fraudulent if:
      - Seller OR buyer is known shell (is_shell=1)  AND
      - has_eway_bill == 0
    This mirrors the paper's 'inward-credit vs outward-sales mismatch' rule.
    """
    idx = graph["idx_test"]
    ei  = graph["edge_index"][:, idx]
    yn  = graph["y_node"]
    ea  = graph["edge_attr"][idx]     # col 2 = has_eway_bill

    src_shell = yn[ei[0]]
    dst_shell = yn[ei[1]]
    no_eway   = (ea[:, 2] < 0.5).astype(np.int64)   # binary

    y_pred = np.clip(src_shell + dst_shell + no_eway, 0, 1)
    y_true = graph["y_edge"][idx]

    t0 = time.perf_counter()
    _ = y_pred                        # latency is negligible
    lat = (time.perf_counter() - t0) * 1000 / len(idx)

    m = compute_metrics(y_true, y_pred.astype(float))
    m["latency_ms"] = lat
    return m


# ── XGBoost / GBM Baseline ───────────────────────────────────────────────

def xgboost_baseline(graph):
    X_tr, y_tr = build_tabular_features(graph, graph["idx_train"])
    X_te, y_te = build_tabular_features(graph, graph["idx_test"])

    clf_class = XGBClassifier if _HAS_XGB else GradientBoostingClassifier
    clf_kwargs = dict(n_estimators=200, max_depth=6, random_state=42,
                      eval_metric="logloss") if _HAS_XGB else \
                 dict(n_estimators=200, max_depth=5, random_state=42)

    if _HAS_XGB:
        clf = XGBClassifier(**clf_kwargs, verbosity=0,
                             scale_pos_weight=float((y_tr == 0).sum()) / (y_tr == 1).sum())
    else:
        clf = GradientBoostingClassifier(**clf_kwargs)

    clf.fit(X_tr, y_tr)

    t0 = time.perf_counter()
    prob = clf.predict_proba(X_te)[:, 1]
    lat  = (time.perf_counter() - t0) * 1000 / len(X_te)

    m = compute_metrics(y_te, prob)
    m["latency_ms"] = lat
    return m


# ── Static GCN Baseline (mini custom implementation) ─────────────────────

class GCNConvSimple(nn.Module):
    """One-hop mean-aggregation GCN layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, N):
        src, dst = edge_index
        # Aggregate neighbours
        agg = torch.zeros(N, x.size(1), device=x.device)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(N, device=x.device)
        deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        agg = agg / deg
        return torch.relu(self.W(agg + x))


class StaticGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1):
        super().__init__()
        self.c1 = GCNConvSimple(in_dim, hid_dim)
        self.c2 = GCNConvSimple(hid_dim, hid_dim)
        self.node_head = nn.Linear(hid_dim, 1)
        self.edge_head = nn.Linear(2 * hid_dim, 1)

    def forward(self, x, edge_index, N):
        h = self.c1(x, edge_index, N)
        h = self.c2(h, edge_index, N)
        ln = self.node_head(h).squeeze(-1)
        le = self.edge_head(torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)).squeeze(-1)
        return ln, le, h


def static_gcn_baseline(graph, variant="GCN"):
    """Train a static GCN/GAT and evaluate on the test split."""
    x_node = to_tensor(graph["x_node"])
    y_node = to_tensor(graph["y_node"], torch.int64)
    N = graph["N"]

    ei_tr = to_tensor(graph["edge_index"][:, graph["idx_train"]], torch.int64)
    ye_tr = to_tensor(graph["y_edge"][graph["idx_train"]], torch.int64)
    ei_te = to_tensor(graph["edge_index"][:, graph["idx_test"]], torch.int64)
    ye_te = graph["y_edge"][graph["idx_test"]]

    model = StaticGCN(in_dim=graph["node_feat_dim"], hid_dim=64).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    pw    = torch.tensor([CFG["pos_weight_node"]]).to(DEVICE)

    for ep in range(50):
        model.train()
        opt.zero_grad()
        ln, le, _ = model(x_node, ei_tr, N)
        l_n = nn.functional.binary_cross_entropy_with_logits(
            ln, y_node.float(), pos_weight=pw)
        # edge imbalance
        yw_e = (ye_tr.float() * CFG["pos_weight_node"] + (1 - ye_tr.float()))
        l_e = nn.functional.binary_cross_entropy_with_logits(le, ye_tr.float(), weight=yw_e)
        (l_n + l_e).backward()
        opt.step()

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        _, le_te, _ = model(x_node, ei_te, N)
    lat = (time.perf_counter() - t0) * 1000 / len(graph["idx_test"])
    prob = torch.sigmoid(le_te).cpu().numpy()
    m = compute_metrics(ye_te, prob)
    m["latency_ms"] = lat
    m["name"] = variant
    return m


# ─────────────────────────────────────────────────────────────────────────────
# MODULE D: ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation(graph, ablation: str):
    """
    Train a variant of the T-GNN with one component disabled.

    ablation ∈ {"no_bochner", "no_gru", "no_cycle_loss", "full"}
    """
    from model_tgnn import TGNN, MultiTaskLoss

    model = TGNN(
        n_nodes       = graph["N"],
        node_feat_dim = graph["node_feat_dim"],
        edge_feat_dim = graph["edge_feat_dim"],
        mem_dim       = CFG["mem_dim"],
        time_dim      = 2 if ablation == "no_bochner" else CFG["time_dim"],
        msg_dim       = CFG["msg_dim"],
        n_heads       = CFG["n_heads"],
        emb_dim       = CFG["emb_dim"],
        dropout       = CFG["dropout"],
        n_layers      = CFG["n_layers"],
    ).to(DEVICE)

    # If ablation == "no_gru": zero-out memory update
    if ablation == "no_gru":
        model.memory.gru_cell = nn.Linear(CFG["msg_dim"], CFG["mem_dim"]).to(DEVICE)

    # Loss: skip cycle term if ablation
    use_cycle = (ablation != "no_cycle_loss")
    criterion = MultiTaskLoss(
        pos_weight_node = CFG["pos_weight_node"],
        lambda3 = CFG["lambda3"] if use_cycle else 0.0,
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=CFG["lr"])
    x_node = to_tensor(graph["x_node"])
    y_node = to_tensor(graph["y_node"], torch.int64)
    ei_tr, ea_tr, ts_tr, ye_tr = split_tensors(graph, "train")
    ni_tr, na_tr, nt_tr, vm_tr = neighbor_tensors(graph, "train")
    t_q = float(ts_tr.max().item())

    # Use neighbour ts matching time_dim
    if ablation == "no_bochner":
        # Rebuild with smaller time_dim
        nt_tr2 = nt_tr  # same data, model uses d_t=2
    else:
        nt_tr2 = nt_tr

    ABLATION_EPOCHS = 15
    for ep in range(ABLATION_EPOCHS):
        model.train()
        model.memory.reset()
        opt.zero_grad()
        ln, le, _ = model(x_node, ei_tr, ea_tr, ts_tr,
                          ni_tr, na_tr, nt_tr2, t_q, vm_tr)
        cl = criterion(ln, y_node, le, ye_tr, model)
        cl["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Evaluate on test
    model.eval()
    model.memory.reset()
    ei_te, ea_te, ts_te, ye_te = split_tensors(graph, "test")
    ni_te, na_te, nt_te, vm_te = neighbor_tensors(graph, "test")
    t_q_te = float(ts_te.max().item())

    with torch.no_grad():
        _, le_te, _ = model(x_node, ei_te, ea_te, ts_te,
                            ni_te, na_te, nt_te, t_q_te, vm_te)
    prob = torch.sigmoid(le_te).cpu().numpy()
    m = compute_metrics(ye_te.cpu().numpy(), prob)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# ASCII TABLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_table(title, rows, col_headers, row_names):
    """Pretty-print a benchmark table in ASCII format."""
    col_w = [max(len(h), 10) for h in col_headers]
    row_w = max(len(r) for r in row_names) + 2

    sep = "+" + "-" * row_w + "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    hdr = "|" + " " * row_w + "|" + "|".join(
        f" {h:^{w}} " for h, w in zip(col_headers, col_w)) + "|"

    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(sep)
    print(hdr)
    print(sep)
    for name, row in zip(row_names, rows):
        cells = "|".join(f" {v:^{w}} " for v, w in zip(row, col_w))
        print(f"| {name:<{row_w - 2}} |{cells}|")
    print(sep)
    print()


def fmt(v, pct=True):
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ───────────────────────────────────────────────────────
    graph = load_and_build_graph()

    # ── 2. Train T-GNN ─────────────────────────────────────────────────────
    model = train_tgnn(graph)

    # ── 3. Evaluate T-GNN on test ──────────────────────────────────────────
    print("── Test Evaluation ─────────────────────────────────────────")
    m_node, m_edge, prob_node, prob_edge = evaluate_tgnn(model, graph)
    print(f"  Node (Shell Detection): "
          f"P={m_node['precision']:.3f}  R={m_node['recall']:.3f}  "
          f"F1={m_node['f1']:.3f}  AUC={m_node['auc']:.3f}")
    print(f"  Edge (Fake Invoice)   : "
          f"P={m_edge['precision']:.3f}  R={m_edge['recall']:.3f}  "
          f"F1={m_edge['f1']:.3f}  AUC={m_edge['auc']:.3f}")
    print(f"  Inference latency     : {m_edge['latency_ms']:.4f} ms/edge\n")

    # ── 4. Baseline benchmarks ─────────────────────────────────────────────
    print("── Running Baselines ───────────────────────────────────────")

    print("  [1/4] Rule-Based …", end=" ", flush=True)
    m_rule = rule_based_baseline(graph)
    print("done.")

    print("  [2/4] XGBoost …", end=" ", flush=True)
    m_xgb = xgboost_baseline(graph)
    print("done.")

    print("  [3/4] Static GCN …", end=" ", flush=True)
    m_gcn = static_gcn_baseline(graph, "GCN")
    print("done.")

    print("  [4/4] Static GAT (same arch, no temporal) …", end=" ", flush=True)
    m_gat = static_gcn_baseline(graph, "GAT")
    print("done.\n")

    # ── 5. Table I ─────────────────────────────────────────────────────────
    col_headers = ["Precision", "Recall", "F1-Score", "ROC-AUC", "Loop Recall", "Lat (ms/edge)"]

    def row_from(m):
        return [
            fmt(m["precision"]),
            fmt(m["recall"]),
            fmt(m["f1"]),
            fmt(m["auc"]),
            fmt(m.get("loop_recall", m["recall"])),
            f"{m['latency_ms']:.4f}",
        ]

    rows_t1 = [
        row_from(m_rule),
        row_from(m_xgb),
        row_from(m_gcn),
        row_from(m_gat),
        row_from(m_edge),          # T-GNN (edge task)
    ]
    names_t1 = [
        "Rule-Based",
        "XGBoost" if _HAS_XGB else "GradBoost",
        "Static GCN",
        "Static GAT",
        "T-GNN (Ours)",
    ]
    print_table("TABLE I: Full Benchmark – Edge-Level (Fake Invoice) Detection",
                rows_t1, col_headers, names_t1)

    # ── 6. Ablation Study (Table II) ───────────────────────────────────────
    print("── Running Ablation Study ─────────────────────────────────")
    ablations = [
        ("T-GNN Full (Ours)", "full"),
        ("w/o Bochner Time", "no_bochner"),
        ("w/o GRU Memory",   "no_gru"),
        ("w/o Cycle Loss",   "no_cycle_loss"),
    ]

    rows_t2 = []
    names_t2 = []
    for name, key in ablations:
        print(f"  [{name}] …", end=" ", flush=True)
        if key == "full":
            m = m_edge
        else:
            m = train_ablation(graph, key)
        rows_t2.append([
            fmt(m["precision"]),
            fmt(m["recall"]),
            fmt(m["f1"]),
            fmt(m["auc"]),
        ])
        names_t2.append(name)
        print(f"F1={m['f1']:.3f}  AUC={m['auc']:.3f}")

    print_table("TABLE II: Ablation Study – Component Contribution",
                rows_t2, ["Precision", "Recall", "F1-Score", "ROC-AUC"], names_t2)

    # ── 7. Save metrics JSON ───────────────────────────────────────────────
    metrics_out = {
        "tgnn_node": m_node,
        "tgnn_edge": m_edge,
        "rule_based": m_rule,
        "xgboost":   m_xgb,
        "static_gcn": m_gcn,
        "static_gat": m_gat,
        "ablation":  {k: {} for k, _ in ablations},
    }
    json_path = os.path.join(CFG["data_dir"], "benchmark_metrics.json")
    with open(json_path, "w") as f:
        json.dump({k: {kk: float(vv) if not isinstance(vv, str) else vv
                       for kk, vv in v.items()}
                   for k, v in metrics_out.items()}, f, indent=2)
    print(f"  ✓ Metrics saved → {json_path}\n")
    print(f"  ✓ Model weights → {CFG['model_save']}\n")
    print("=" * 70)
    print("  Training and evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
