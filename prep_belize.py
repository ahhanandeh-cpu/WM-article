"""Build data/units_belize.csv from the Cefas CLiP Belize household file (different layout from the 2018 files).
usage: python3 prep_belize.py "Household solid waste audit Belize data.csv"
Uses the original 'Value' (kg per sample), not the 'Correted value' rates; household samples only."""
import sys, os
import pandas as pd
from datasets import DATA, F7

M = {"organics": "Organics", "paper_and_cardboard": "Paper", "plastic": "Plastics", "metal": "Metals",
     "glass_glass": "Glass", "hygiene": "Hygiene"}
a = pd.read_csv(sys.argv[1], low_memory=False, encoding_errors="replace")
a = a[(a.Premises_type == "pt_household") & (a.Measure == "weight")]
a["F"] = a.Material.map(lambda m: None if m == "total_sample" or pd.isna(m) else M.get(m, "Other"))
U = a[a.F.notna()].pivot_table(index="ID", columns="F", values="Value", aggfunc="sum").fillna(0)
U["sum"] = U[F7].sum(axis=1)
U["total"] = a[a.Material == "total_sample"].groupby("ID").Value.sum()
g = a.groupby("ID"); U["site"] = g.Site.first(); U["type"] = g.Type.first()
U["collection_interval_days"] = g["Collection frequency"].first()
U.to_csv(os.path.join(DATA, "units_belize.csv")); print(len(U), "households")
