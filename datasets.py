"""Loaders: one Population per fully sorted household audit.  Organics is always the first fraction."""
import os
import numpy as np, pandas as pd
from rbys import Population

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
F7 = ["Organics", "Paper", "Plastics", "Metals", "Glass", "Hygiene", "Other"]

# Cefas CLiP item 'Material' -> fraction (the 49 sorted items are grouped by their Material label)
CLIP_MAP = {"Organics": "Organics", "Paper and cardboard": "Paper", "Plastic": "Plastics", "Metal": "Metals",
            "Glass": "Glass", "Hygiene": "Hygiene"}


def load_solomon():
    """174 two-day household samples (Guadalcanal + Malaita), CLiP 2018.  data/units.csv is built by prep_data.py."""
    U = pd.read_csv(os.path.join(DATA, "units_solomon.csv"), index_col=0)
    P = U[U.index.str.match(r"^(AH|AR|R)\d") & (U.mult == 2)].copy()
    rec = {}
    f = os.path.join(DATA, "interviews.csv")
    if os.path.exists(f):
        I = pd.read_csv(f).rename(columns={"Household/Commercial/Ship_ID": "ID"}).set_index("ID")
        rec["HHSIZE"] = P.join(I[["Number_of_people_in_household"]])["Number_of_people_in_household"].to_numpy(float)
    W = P["sum"].to_numpy()
    scale = np.where(P.total.to_numpy() > 0, P.total.to_numpy(), W)
    site = P.site.map(lambda s: "Guadalcanal" if "Lunga" in s or s.startswith("Guadal") else s).to_numpy()
    return Population("Solomon Islands", F7, W, P[F7].to_numpy(), scale, rec, strata=P.site.to_numpy())


def load_kampala():
    """103 households, 7-day collection, three income parishes (Katukiza et al. 2026; openwashdata, CC BY 4.0)."""
    K = pd.read_csv(os.path.join(DATA, "kampala.csv"))
    fr = pd.DataFrame({
        "Organics": K.waste_food_kg + K.waste_garden_kg + K.waste_wood_kg,
        "Paper": K.waste_paper_kg,
        "Plastics": K.waste_polythene_kg + K.waste_plastics_kg,
        "Metals": K.waste_metals_kg,
        "Glass": K.waste_glass_kg,
        "Other": K.waste_textiles_kg + K.waste_other_kg,
    })
    W = fr.sum(axis=1).to_numpy()
    rec = {"HHSIZE": K.occupants.to_numpy(float), "DENSITY": (W / K.volume_l).to_numpy(float)}
    return Population("Kampala", list(fr.columns), W, fr.to_numpy(), W, rec, strata=K.parish.to_numpy())


def clip_units(abundance_csv, id_col=None):
    """Fraction masses per sample from a Cefas CLiP abundance file (same structure across countries)."""
    a = pd.read_csv(abundance_csv, low_memory=False, encoding_errors="replace")
    a.columns = [c.strip() for c in a.columns]
    id_col = id_col or next(c for c in a.columns if "Household" in c and "ID" in c)
    a = a.rename(columns={id_col: "ID"})
    w = a[a.Measure.astype(str).str.strip().str.lower() == "weight"].copy()
    w["F"] = w.Material.map(lambda m: None if pd.isna(m) else CLIP_MAP.get(str(m).strip(), "Other"))
    U = w[w.F.notna()].pivot_table(index="ID", columns="F", values="Value", aggfunc="sum").fillna(0)
    for f in F7:
        if f not in U: U[f] = 0.0
    U["sum"] = U[F7].sum(axis=1)
    tot = w[w.Item.astype(str).str.contains("Total sample weight", case=False)]
    U["total"] = tot.groupby("ID").Value.sum()
    U["site"] = w.groupby("ID").Site.first()
    U["mult"] = w.groupby("ID").Multiplier.first()
    return U


def _clip_population(name, units_csv, interviews_csv, household_regex):
    U = pd.read_csv(os.path.join(DATA, units_csv), index_col=0)
    P = U[U.index.str.match(household_regex)].copy()
    rec = {}
    f = os.path.join(DATA, interviews_csv)
    if os.path.exists(f):
        I = pd.read_csv(f, low_memory=False)
        I = I.rename(columns={I.columns[0]: "ID"}).drop_duplicates("ID").set_index("ID")
        rec["HHSIZE"] = P.join(I[["Number_of_people_in_household"]])["Number_of_people_in_household"].to_numpy(float)
    W = P["sum"].to_numpy()
    tot = P.total.to_numpy(float)
    scale = np.where(np.isfinite(tot) & (tot > 0), tot, W)       # recorded scale reading before sorting
    return Population(name, F7, W, P[F7].to_numpy(), scale, rec, strata=P.site.to_numpy()), P


def load_vanuatu():
    """201 household samples (Port Vila, Lelepa: 2 days of waste; Luganville, Ifira, Shefa rural: 1 day), CLiP 2018."""
    return _clip_population("Vanuatu", "units_vanuatu.csv", "vanuatu_household_size.csv", r"^(H|IFI|LH|LV|SHR) ?\d")[0]


LOADERS = {"solomon": load_solomon, "vanuatu": load_vanuatu, "kampala": load_kampala}


def load_belize():
    """209 household samples with sorted weights in 8 communities, CLiP 2019 (Cefas 2020; data in kg).
    data/units_belize.csv is built by prep_belize.py; household sizes above 30 are treated as entry errors."""
    U = pd.read_csv(os.path.join(DATA, "units_belize.csv"), index_col=0)
    S = pd.read_csv(os.path.join(DATA, "belize_household_size.csv")).set_index("ID")
    rec = {"HHSIZE": U.join(S)["House_number_people"].to_numpy(float)}
    W = U["sum"].to_numpy()
    tot = U.total.to_numpy(float)
    scale = np.where(np.isfinite(tot) & (tot > 0), tot, W)
    return Population("Belize", F7, W, U[F7].to_numpy(), scale, rec, strata=U.site.to_numpy())


LOADERS["belize"] = load_belize
