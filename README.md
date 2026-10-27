# rank-before-you-sort

Ranked set sampling (RSS) tools for municipal solid waste composition audits.

This repository accompanies the article

> Hanandeh, A. A. *Rank before you sort: a ranked set sampling protocol for cheaper and safer municipal
> solid waste composition audits, validated on household audits from four countries.* (submitted to *Waste Management*).

In the protocol, each unit (a household bag, a bin, a sub-sample of a vehicle load) is first weighed, photographed and
given a visual organic score. Units are then ranked within small sets by this cheap proxy, and only one unit per set
is hand-sorted. The repository contains:

| | |
|---|---|
| `rbys_tool.py` | **Field tool.** Plan an audit from a pilot, print recording sheets, and estimate a composition with confidence intervals |
| `rbys.py` | Resampling engine: audit emulation, planning rule, pilot decision rule, household bootstrap, stratified RSS |
| `datasets.py`, `prep_clip.py`, `prep_belize.py` | Loaders for the four validation populations; builders from the Cefas CLiP audit files |
| `run_all.py`, `make_tables.py`, `make_figures.py` | Reproduce the multi-country validation (Tables 2, 4, 5, S5–S12; Figs 2–3) |
| `simulation/` | Simulation study (Table 3, Figs S1–S3, Tables S1–S2) |
| `data/` | Derived per-household fraction masses (see `data/README.md` for sources and licences) |
| `examples/` | Example pilot and audit files for the field tool |
| `results/` | Results files as used in the article |
| `legacy/` | Scripts of the first submission (Solomon Islands only), kept for the record |

## Installation

Python ≥ 3.10 with NumPy, SciPy, pandas and matplotlib:

```
pip install -r requirements.txt
```

## Using the field tool

**1. Pilot.** Sort about 20 randomly chosen units. For each unit, record the scale weight, the visual score
(0–5, see Supplementary Section S4 of the article), any other cheap proxy, and the sorted fraction masses.
Then run:

```
python3 rbys_tool.py plan examples/pilot.csv --mass W \
    --fractions Organics,Paper,Plastics,Metals,Glass,Other \
    --proxies W,S,household_size --score S --m 3 --cost-ratio 0.03 --priority Organics=1
```

The tool reports, for each proxy and fraction, the normal-score correlation, the predicted relative efficiency
(adjusted for pilot size), and the cost-aware choice between ranking and simple random sampling.
`--priority` sets the policy weights, for example `Organics=1,Metals=0.5,Plastics=0.5`. `--cost-ratio` is
inspection time divided by sorting time per unit.

**2. Recording sheet.**

```
python3 rbys_tool.py sheet --m 3 --r 10 --seed 7 > sheet.csv
```

**3. Estimate.** Enter one row per sorted unit with its cycle, its rank (= set number) and its fraction masses:

```
python3 rbys_tool.py estimate examples/audit.csv --mass W --fractions Organics,Paper,Plastics,Metals,Glass,Other
```

The output gives each share with its standard error and a t-interval with m(r−1) degrees of freedom. It flags
fractions absent from most units, for which no interval is reliable.

The example files are drawn from the Kampala data with an *emulated* visual score. They illustrate the file
format only.

## Reproducing the article

```
bash reproduce.sh            # full run: a few minutes for the validation; the simulation study takes longer
bash reproduce.sh --quick    # smoke test with fewer replicates
```

This writes `results/*.json`, `tables/*.tex` and `figures/*.pdf`. Random seeds are fixed, so the full run
reproduces the published numbers up to the stated Monte Carlo error.

To rebuild `data/units_*.csv` from the original Cefas files, download the *Abundance* CSV of each 2018 audit
(see `data/README.md`) and run `python3 prep_clip.py <abundance.csv> <country>`. For Belize, run
`python3 prep_belize.py "Household solid waste audit Belize data.csv"`.

## Citation

Please cite the article and this archive (see `CITATION.cff`). Please also cite the original data (see `data/README.md`).

## Licence

The code is licensed under MIT. The files in `data/` keep the licences of their sources.
