"""Build data/units_<country>.csv (one row per sorted sample, fraction masses in kg) from a Cefas CLiP
abundance file.  The CLiP files list every sorted item (e.g. 49 items in the Solomon Islands file) with
its 'Material'; items are grouped into seven fractions by that label (see datasets.CLIP_MAP).

usage:  python3 prep_clip.py <abundance.csv> <country>
"""
import sys
from datasets import clip_units, DATA
import os

src, country = sys.argv[1], sys.argv[2].lower()
U = clip_units(src)
out = os.path.join(DATA, f"units_{country}.csv")
U.to_csv(out)
print(f"{len(U)} samples -> {out}")
print(U.site.value_counts().to_string())
