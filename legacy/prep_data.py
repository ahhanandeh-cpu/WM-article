"""Build data/units.csv (one row per sorted sample, fraction masses in kg) from the Cefas abundance file."""
import pandas as pd
a = pd.read_csv("data/abundance.csv", low_memory=False).rename(columns={"Household/Commercial/Ship ID": "ID"})
w = a[a.Measure == "Weight"].copy()
MAP = {"Organics": "Organics", "Paper and cardboard": "Paper", "Plastic": "Plastics", "Metal": "Metals",
       "Glass": "Glass", "Hygiene": "Hygiene"}
w["F"] = w.Material.map(lambda m: None if pd.isna(m) else MAP.get(m, "Other"))
U = w[w.F.notna()].pivot_table(index="ID", columns="F", values="Value", aggfunc="sum").fillna(0)
U["sum"] = U.sum(axis=1)
U["total"] = w[w.Item == "Total sample weight (KG)"].set_index("ID").Value
U["site"] = w.groupby("ID").Site.first(); U["mult"] = w.groupby("ID").Multiplier.first()
U.to_csv("data/units_solomon.csv")
