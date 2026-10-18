import datetime
import random
import networkx as nx
import numpy as np
import pandas as pd

# Reproducibility ke liye seed
random.seed(42)
np.random.seed(42)

# Configuration
NUM_LEGIT_COMPANIES = 1000
NUM_SHELL_COMPANIES = 80
NUM_LEGIT_INVOICES = 5000
STATE_CODES = ["07", "27", "29", "06", "19", "33", "24", "09", "10", "36"]


# 1. Helper: GSTIN Generator
def make_gstin():
  state = random.choice(STATE_CODES)
  pan_alpha = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
  pan_num = f"{random.randint(1000, 9999)}"
  pan_last = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
  return f"{state}{pan_alpha}{pan_num}{pan_last}1Z{random.choice('123456789ABCDEF')}"


# 2. Generate Companies (Nodes)
companies = []
base_date = datetime.date(2025, 1, 1)

# Legitimate Companies
for i in range(NUM_LEGIT_COMPANIES):
  reg_date = base_date - datetime.timedelta(days=random.randint(200, 1800))
  companies.append({
      "gstin": make_gstin(),
      "company_name": f"Enterprise_Genuine_{i:04d}",
      "registration_date": reg_date.isoformat(),
      "capital_lakhs": round(random.uniform(25.0, 500.0), 2),
      "employee_count": random.randint(10, 150),
      "business_type": random.choice(
          ["Manufacturing", "Wholesale", "Retail", "Logistics"]
      ),
      "compliance_score": round(random.uniform(7.0, 9.9), 1),
      "is_shell": 0,
  })

# Shell Companies
for i in range(NUM_SHELL_COMPANIES):
  reg_date = base_date - datetime.timedelta(days=random.randint(10, 50))
  companies.append({
      "gstin": make_gstin(),
      "company_name": f"Trading_Shell_{i:03d}",
      "registration_date": reg_date.isoformat(),
      "capital_lakhs": round(random.uniform(1.0, 4.0), 2),
      "employee_count": random.randint(1, 3),
      "business_type": "Trading_Intermediary",
      "compliance_score": round(random.uniform(2.0, 4.5), 1),
      "is_shell": 1,
  })

df_companies = pd.DataFrame(companies)
legit_gstins = df_companies[df_companies["is_shell"] == 0]["gstin"].tolist()
shell_gstins = df_companies[df_companies["is_shell"] == 1]["gstin"].tolist()

# 3. Generate Invoices (Dynamic Edges)
invoices = []
inv_counter = 100001
start_time = datetime.datetime(2025, 1, 1, 9, 0, 0)

# A. Genuine Transactions
for _ in range(NUM_LEGIT_INVOICES):
  seller, buyer = random.sample(legit_gstins, 2)
  taxable = round(random.uniform(15000, 2000000), 2)
  rate = random.choice([0.05, 0.12, 0.18, 0.28])
  gst_amt = round(taxable * rate, 2)

  tx_time = start_time + datetime.timedelta(
      seconds=random.randint(0, 180 * 86400)
  )

  invoices.append({
      "invoice_no": f"INV{inv_counter}",
      "seller_gstin": seller,
      "buyer_gstin": buyer,
      "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
      "taxable_amount": taxable,
      "gst_rate": rate,
      "gst_amount": gst_amt,
      "total_amount": round(taxable + gst_amt, 2),
      "has_eway_bill": 1 if random.random() > 0.04 else 0,
      "fraud_category": "Normal",
      "is_fraud": 0,
  })
  inv_counter += 1

# B. Inject Circular Trading Loops (10 Closed Cycles)
for ring_id in range(10):
  cycle_size = random.choice([3, 4, 5, 6])
  ring_nodes = random.sample(shell_gstins, cycle_size)

  circulating_val = round(random.uniform(6000000, 30000000), 2)
  ring_time = start_time + datetime.timedelta(
      days=random.randint(15, 150), hours=random.randint(10, 16)
  )

  for hop in range(cycle_size):
    u = ring_nodes[hop]
    v = ring_nodes[(hop + 1) % cycle_size]

    hop_val = round(circulating_val * random.uniform(0.99, 1.01), 2)
    gst_amt = round(hop_val * 0.18, 2)
    hop_time = ring_time + datetime.timedelta(
        minutes=hop * random.randint(10, 25)
    )

    invoices.append({
        "invoice_no": f"INV{inv_counter}",
        "seller_gstin": u,
        "buyer_gstin": v,
        "timestamp": hop_time.strftime("%Y-%m-%d %H:%M:%S"),
        "taxable_amount": hop_val,
        "gst_rate": 0.18,
        "gst_amount": gst_amt,
        "total_amount": round(hop_val + gst_amt, 2),
        "has_eway_bill": 0,
        "fraud_category": f"Circular_Loop_{ring_id+1}",
        "is_fraud": 1,
    })
    inv_counter += 1

# C. Inject Fake Invoice Mills
for mill_id in range(4):
  master = shell_gstins[mill_id * 6]
  recipients = random.sample(legit_gstins, 12)
  burst_time = start_time + datetime.timedelta(
      days=random.randint(20, 160), hours=23
  )

  for idx, rec in enumerate(recipients):
    val = round(random.uniform(700000, 2500000), 2)
    gst_amt = round(val * 0.18, 2)
    t_stamp = burst_time + datetime.timedelta(minutes=idx * 2)

    invoices.append({
        "invoice_no": f"INV{inv_counter}",
        "seller_gstin": master,
        "buyer_gstin": rec,
        "timestamp": t_stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "taxable_amount": val,
        "gst_rate": 0.18,
        "gst_amount": gst_amt,
        "total_amount": round(val + gst_amt, 2),
        "has_eway_bill": 0,
        "fraud_category": "Fake_Invoice_Mill",
        "is_fraud": 1,
    })
    inv_counter += 1

# Sort Chronologically and Save
df_invoices = pd.DataFrame(invoices)
df_invoices["dt"] = pd.to_datetime(df_invoices["timestamp"])
df_invoices = df_invoices.sort_values(by="dt").drop(columns=["dt"])

df_companies.to_csv("gst_companies.csv", index=False)
df_invoices.to_csv("gst_invoices.csv", index=False)
print("Files successfully generated: gst_companies.csv and gst_invoices.csv")