"""
=============================================================================
explain_subgraph.py  —  GNNExplainer Subgraph Audit for GST Fraud
=============================================================================
Paper: "Temporal Graph Neural Networks for Real-Time Detection of
         GST Circular Trading and Shell Entities"

For every node with predicted shell-entity score  ŷ_i > 0.80 the script:
  1. Extracts a 2-3-hop ego subgraph centred on that node.
  2. Applies a gradient-based attribution (GNNExplainer proxy) to score
     which edges and neighbours most contributed to the fraud prediction.
  3. Serialises a human-readable Audit Dictionary (JSON) for each suspect.

Output
------
  audit_reports/
    audit_<GSTIN>.json      – per-node audit file
    audit_summary.json      – consolidated summary of all suspects
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model_tgnn import TGNN
from train_and_evaluate import (
    load_and_build_graph, to_tensor, split_tensors,
    neighbor_tensors, CFG, build_model,
)

DEVICE = torch.device(CFG["device"])
THRESHOLD_SHELL = 0.80       # flag nodes above this score
MAX_HOPS        = 2          # 2-3 hop neighbourhood
TOP_K_EDGES     = 5          # top-K influential edges to report

AUDIT_DIR = os.path.join(CFG["data_dir"], "audit_reports")
os.makedirs(AUDIT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GNNExplainer  (gradient × activation attribution)
# ─────────────────────────────────────────────────────────────────────────────

class GNNExplainerProxy:
    """
    Lightweight gradient-based GNN explainability for the T-GNN.

    Implements the GNNExplainer masking approach:
      - Optimise a continuous edge-mask M over the neighbourhood so that
        the masked model output matches the original.
      - Report top-K edges with highest mask values as the explanation.

    For speed and compatibility we use gradient × input attribution
    (one-shot, no iterative optimisation) which is a validated proxy.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def explain_node(
        self,
        target_node: int,
        graph: dict,
        n_hops: int = MAX_HOPS,
    ) -> dict:
        """
        Parameters
        ----------
        target_node : int node id
        graph       : full graph dict (from load_and_build_graph)
        n_hops      : neighbourhood hop radius

        Returns
        -------
        explanation dict
        """
        N  = graph["N"]
        ei = graph["edge_index"]        # (2, E_total)
        ea = graph["edge_attr"]         # (E_total, d_edge)
        ts = graph["timestamps"]        # (E_total,)

        # ── 1. Extract K-hop neighbourhood ────────────────────────────────
        neigh_nodes = {target_node}
        neigh_edges_idx = []

        frontier = {target_node}
        for _ in range(n_hops):
            new_frontier = set()
            for e_idx in range(ei.shape[1]):
                src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
                if src in frontier or dst in frontier:
                    neigh_nodes.add(src)
                    neigh_nodes.add(dst)
                    neigh_edges_idx.append(e_idx)
                    new_frontier.update([src, dst])
            frontier = new_frontier - neigh_nodes
            neigh_nodes |= new_frontier

        if not neigh_edges_idx:
            return {"node_id": target_node, "note": "isolated node", "edges": []}

        neigh_edges_idx = list(set(neigh_edges_idx))

        # ── 2. Build local tensors with grad-tracking ──────────────────────
        ea_local = torch.tensor(ea[neigh_edges_idx], dtype=torch.float32,
                                device=DEVICE, requires_grad=True)
        ei_local = to_tensor(ei[:, neigh_edges_idx], torch.int64)
        ts_local = to_tensor(ts[neigh_edges_idx])

        # Use the test-split neighbour tensors as background
        ni_te = to_tensor(graph["ni_test"], torch.int64)
        na_te = to_tensor(graph["na_test"])
        nt_te = to_tensor(graph["nt_test"])
        vm_te = to_tensor(graph["vm_test"], torch.bool)
        x_node = to_tensor(graph["x_node"])

        t_q = float(ts_local.max().item())

        self.model.eval()
        self.model.memory.reset()

        # ── 3. Forward pass (track gradients w.r.t. edge features) ────────
        logit_node, logit_edge, z = self.model(
            x_node, ei_local, ea_local, ts_local,
            ni_te, na_te, nt_te, t_q, vm_te,
        )

        # Scalar target: shell probability of the target node
        target_logit = logit_node[target_node]
        target_logit.backward()

        # Gradient × |feature| attribution score per edge
        with torch.no_grad():
            grad     = ea_local.grad.abs()          # (E_local, d_edge)
            ea_vals  = ea_local.abs()               # (E_local, d_edge)
            edge_imp = (grad * ea_vals).sum(dim=-1) # (E_local,)
            edge_imp = edge_imp.cpu().numpy()

        # ── 4. Select top-K edges ──────────────────────────────────────────
        top_k = min(TOP_K_EDGES, len(neigh_edges_idx))
        top_idx = np.argsort(edge_imp)[::-1][:top_k]
        top_edge_global = [neigh_edges_idx[i] for i in top_idx]
        top_scores      = edge_imp[top_idx]

        return {
            "node_id":        target_node,
            "top_edge_indices": top_edge_global,
            "top_edge_scores":  top_scores.tolist(),
            "neighbourhood_size": {
                "nodes": len(neigh_nodes),
                "edges": len(neigh_edges_idx),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_audit_report(
    node_id:   int,
    gstin:     str,
    prob_shell: float,
    explanation: dict,
    graph:     dict,
    df_inv:    pd.DataFrame,
    df_comp:   pd.DataFrame,
    id2gstin:  dict,
) -> dict:
    """
    Construct a human-readable audit dictionary for one flagged entity.
    """
    # ── Node-level info ────────────────────────────────────────────────────
    comp_row = df_comp[df_comp["gstin"] == gstin]
    if len(comp_row) > 0:
        comp_info = comp_row.iloc[0].to_dict()
    else:
        comp_info = {"gstin": gstin, "note": "not in company master"}

    # ── Counterparty GSTINs (1-hop) ────────────────────────────────────────
    ei = graph["edge_index"]
    ea = graph["edge_attr"]
    ts = graph["timestamps"]

    src_edges = np.where(ei[0] == node_id)[0]
    dst_edges = np.where(ei[1] == node_id)[0]

    counterparty_ids = set(ei[1, src_edges].tolist()) | set(ei[0, dst_edges].tolist())
    counterparty_ids.discard(node_id)
    counterparty_gstins = [id2gstin.get(i, str(i)) for i in counterparty_ids]

    # ── Detect circular trading involvement ────────────────────────────────
    circular_invoices = []
    if "fraud_category" in df_inv.columns:
        inv_rows = df_inv[
            (df_inv["seller_gstin"] == gstin) | (df_inv["buyer_gstin"] == gstin)
        ]
        circ_rows = inv_rows[inv_rows["fraud_category"].str.startswith("Circular", na=False)]
        for _, r in circ_rows.iterrows():
            circular_invoices.append({
                "invoice_no":      r["invoice_no"],
                "from":            r["seller_gstin"],
                "to":              r["buyer_gstin"],
                "timestamp":       r["timestamp"],
                "taxable_amount":  float(r["taxable_amount"]),
                "gst_rate":        float(r["gst_rate"]),
                "has_eway_bill":   int(r["has_eway_bill"]),
                "fraud_category":  r["fraud_category"],
            })

    # ── Cycle execution timeline ───────────────────────────────────────────
    cycle_timeline = []
    if circular_invoices:
        circ_df = pd.DataFrame(circular_invoices).sort_values("timestamp")
        t_first = pd.to_datetime(circ_df["timestamp"].iloc[0])
        t_last  = pd.to_datetime(circ_df["timestamp"].iloc[-1])
        duration_hrs = (t_last - t_first).total_seconds() / 3600.0

        total_in  = circ_df[circ_df["to"] == gstin]["taxable_amount"].sum()
        total_out = circ_df[circ_df["from"] == gstin]["taxable_amount"].sum()
        turnover_discrepancy = abs(total_in - total_out)

        cycle_timeline = {
            "first_transaction": str(t_first),
            "last_transaction":  str(t_last),
            "cycle_duration_hrs": round(duration_hrs, 4),
            "total_inward_txbl":  round(float(total_in), 2),
            "total_outward_txbl": round(float(total_out), 2),
            "value_addition_net": round(float(total_in - total_out), 2),
            "turnover_discrepancy": round(float(turnover_discrepancy), 2),
        }

    # ── Missing e-Way bill analysis ────────────────────────────────────────
    no_eway_count = 0
    for e_idx in range(ei.shape[1]):
        if ei[0, e_idx] == node_id or ei[1, e_idx] == node_id:
            if ea[e_idx, 2] < 0.5:  # has_eway_bill col
                no_eway_count += 1

    # ── Top influential edges (from explainer) ─────────────────────────────
    top_edges_detail = []
    for e_idx, score in zip(explanation.get("top_edge_indices", []),
                            explanation.get("top_edge_scores", [])):
        if e_idx < len(df_inv):
            row = df_inv.iloc[e_idx]
            top_edges_detail.append({
                "invoice_no":     row["invoice_no"],
                "from":           row["seller_gstin"],
                "to":             row["buyer_gstin"],
                "timestamp":      row["timestamp"],
                "taxable_amount": float(row["taxable_amount"]),
                "is_fraud":       int(row["is_fraud"]),
                "influence_score": round(float(score), 6),
            })

    report = {
        "audit_meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":        "T-GNN (Temporal Graph Neural Network)",
            "threshold":    THRESHOLD_SHELL,
        },
        "flagged_entity": {
            "node_id":          node_id,
            "gstin":            gstin,
            "predicted_shell_probability": round(float(prob_shell), 4),
            "company_info":     {k: str(v) for k, v in comp_info.items()},
        },
        "network_summary": {
            "counterparty_gstins":       counterparty_gstins,
            "counterparty_count":        len(counterparty_gstins),
            "transactions_without_ewaybill": no_eway_count,
            "neighbourhood_size_2hop":   explanation.get("neighbourhood_size", {}),
        },
        "circular_trading_evidence": {
            "circular_invoices_found":   len(circular_invoices),
            "circular_invoices":         circular_invoices[:10],  # cap for readability
            "cycle_execution_timeline":  cycle_timeline,
        },
        "gnn_explanation": {
            "method": "Gradient × Activation Attribution (GNNExplainer proxy)",
            "top_influential_edges": top_edges_detail,
        },
        "risk_assessment": {
            "risk_level": "HIGH" if prob_shell >= 0.90 else "MEDIUM-HIGH",
            "flags": [
                f"Predicted shell probability: {prob_shell:.1%}",
                f"Found {len(circular_invoices)} circular trading invoices.",
                f"Missing e-Way bills on {no_eway_count} transactions.",
                f"Connected to {len(counterparty_gstins)} counterparties.",
            ],
            "recommended_action": "Immediate field audit and GSTIN suspension recommended."
            if prob_shell >= 0.90 else "Schedule detailed desk audit within 30 days.",
        },
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load graph and model ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GNNExplainer Audit Trail Generator")
    print(f"{'='*70}\n")

    graph  = load_and_build_graph()
    model  = build_model(graph)

    ckpt_path = CFG["model_save"]
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No saved model at {ckpt_path}")
        print("  Please run train_and_evaluate.py first.\n")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  ✓ Loaded model weights from {ckpt_path}\n")

    # ── 2. Load raw DataFrames for metadata ────────────────────────────────
    data_dir = CFG["data_dir"]
    df_comp  = pd.read_csv(os.path.join(data_dir, "gst_companies.csv"))
    df_inv   = pd.read_csv(os.path.join(data_dir, "gst_invoices.csv"))
    df_inv["dt"] = pd.to_datetime(df_inv["timestamp"])
    df_inv = df_inv.sort_values("dt").reset_index(drop=True)

    id2gstin = {v: k for k, v in graph["gstin2id"].items()}

    # ── 3. Run T-GNN on full graph to get predictions ──────────────────────
    model.memory.reset()
    x_node = to_tensor(graph["x_node"])
    ei_te, ea_te, ts_te, ye_te = split_tensors(graph, "test")
    ni_te, na_te, nt_te, vm_te = neighbor_tensors(graph, "test")
    t_q = float(ts_te.max().item())

    with torch.no_grad():
        logit_node, logit_edge, z = model(
            x_node, ei_te, ea_te, ts_te,
            ni_te, na_te, nt_te, t_q, vm_te
        )

    prob_node = torch.sigmoid(logit_node).cpu().numpy()  # (N,)
    prob_edge = torch.sigmoid(logit_edge).cpu().numpy()  # (E_test,)

    # ── 4. Filter high-risk nodes ──────────────────────────────────────────
    flagged_nodes = np.where(prob_node >= THRESHOLD_SHELL)[0]
    print(f"  Nodes flagged (ŷ ≥ {THRESHOLD_SHELL}): {len(flagged_nodes)}")

    if len(flagged_nodes) == 0:
        # Lower threshold to capture at least a few examples
        THRESHOLD_SHELL_ADJ = 0.50
        flagged_nodes = np.where(prob_node >= THRESHOLD_SHELL_ADJ)[0][:10]
        print(f"  [WARN] Lowering threshold to {THRESHOLD_SHELL_ADJ} "
              f"→ {len(flagged_nodes)} nodes flagged.\n")

    # Sort by descending probability
    flagged_nodes = sorted(flagged_nodes, key=lambda i: -prob_node[i])

    # ── 5. Generate explanations ───────────────────────────────────────────
    explainer = GNNExplainerProxy(model)
    summary   = []

    print(f"\n  Generating audit reports for top flagged nodes …\n")
    for rank, node_id in enumerate(flagged_nodes[:20], start=1):   # cap at 20
        gstin      = id2gstin.get(int(node_id), f"UNK_{node_id}")
        p_shell    = float(prob_node[node_id])

        print(f"  [{rank:02d}] {gstin}  p_shell={p_shell:.4f}", end="  ")

        try:
            expl = explainer.explain_node(int(node_id), graph)
        except Exception as e:
            print(f"[EXPLAIN ERROR: {e}]")
            expl = {"node_id": int(node_id), "top_edge_indices": [],
                    "top_edge_scores": [], "neighbourhood_size": {}}

        report = build_audit_report(
            node_id    = int(node_id),
            gstin      = gstin,
            prob_shell = p_shell,
            explanation = expl,
            graph      = graph,
            df_inv     = df_inv,
            df_comp    = df_comp,
            id2gstin   = id2gstin,
        )

        # Save individual report
        fname = os.path.join(AUDIT_DIR, f"audit_{gstin.replace('/', '_')}.json")
        with open(fname, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"→ {os.path.basename(fname)}")

        summary.append({
            "rank":          rank,
            "gstin":         gstin,
            "node_id":       int(node_id),
            "shell_prob":    round(p_shell, 4),
            "risk_level":    report["risk_assessment"]["risk_level"],
            "circular_invoices_found": report["circular_trading_evidence"]["circular_invoices_found"],
            "counterparty_count": report["network_summary"]["counterparty_count"],
            "missing_ewaybill_count": report["network_summary"]["transactions_without_ewaybill"],
            "audit_file":    fname,
        })

    # ── 6. Save consolidated summary ──────────────────────────────────────
    summary_path = os.path.join(AUDIT_DIR, "audit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "total_flagged": len(flagged_nodes),
             "reports_generated": len(summary),
             "threshold": THRESHOLD_SHELL,
             "suspects": summary},
            f, indent=2
        )

    # ── 7. Print summary table ────────────────────────────────────────────
    sep = "+" + "-"*18 + "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*15 + "+" + "-"*12 + "+"
    hdr = f"| {'GSTIN':<16} | {'Shell Prob':>10} | {'Risk Level':>12} | {'Circ Invoices':>13} | {'Missing EWB':>10} |"
    print(f"\n{'─'*75}")
    print("  AUDIT SUMMARY — High-Risk Shell Entity Report")
    print(f"{'─'*75}")
    print(sep)
    print(hdr)
    print(sep)
    for s in summary[:10]:
        print(f"| {s['gstin']:<16} | {s['shell_prob']:>10.4f} | {s['risk_level']:>12} "
              f"| {s['circular_invoices_found']:>13} | {s['missing_ewaybill_count']:>10} |")
    print(sep)
    print(f"\n  ✓ {len(summary)} audit reports saved → {AUDIT_DIR}/")
    print(f"  ✓ Summary         saved → {summary_path}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
