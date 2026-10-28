# Data provenance

All files hold one row per fully sorted household sample, with fraction masses in kg. Each row is derived from
openly licensed data. Please cite the original sources.

| File | Source | Licence |
|---|---|---|
| `units_solomon.csv` | Cefas (2020). *2018 CLiP Solomon Islands land waste audits and interviews.* Cefas Data Portal record 19858. | Open Government Licence v3.0 |
| `units_vanuatu.csv`, `vanuatu_household_size.csv` | Cefas (2019). *CLiP Vanuatu land waste audits and interviews.* doi:10.14466/CefasDataHub.77 | Open Government Licence v3.0 |
| `units_belize.csv`, `belize_household_size.csv` | Cefas (2020). *CLiP Belize land waste audits and interviews 2019.* doi:10.14466/CefasDataHub.97 | Open Government Licence v3.0 |
| `kampala.csv` | Katukiza et al. (2026), *Front. Environ. Sci.* 14, doi:10.3389/fenvs.2026.1889921; data package openwashdata/solidwastekampala, doi:10.5281/zenodo.21519796 | CC BY 4.0 |

These files contain public sector information licensed under the Open Government Licence v3.0.

## Derivation

* **CLiP files** (`units_solomon.csv`, `units_vanuatu.csv`) are built by `prep_clip.py` from the *Abundance* file of each
  audit. Rows with `Measure == "Weight"` are summed within the `Material` label into seven fractions: Organics; Paper
  (paper and cardboard, incl. liquid paperboard and cigarette butts); Plastics; Metals; Glass; Hygiene; Other
  (all remaining labels). Further columns:
  * `sum`: the sum of the sorted fractions (the unit mass used for estimation);
  * `total`: the scale reading recorded before sorting (used as the weight proxy);
  * `site`: the audit site;
  * `mult`: the number of days of waste in the sample.
  Commercial and ship samples are kept in the files; the loaders in `datasets.py` select household samples.
* **`units_belize.csv`** is built by `prep_belize.py` from *Household solid waste audit Belize data.csv*. That file has a different
  layout from the 2018 files (lower-case material labels, numeric IDs, `Premises_type`). The original `Value` column (kg per sample) is used, not the
  derived weekly rates in `Correted value`, and household samples only. `collection_interval_days` is the interval reported in the interview.
* **`belize_household_size.csv`** keeps the household ID and `House_number_people`. One entry of 1,620 people is set to missing.
* **`vanuatu_household_size.csv`** keeps only the household ID and the number of people in the household from the
  interview file. Other interview fields, including GPS coordinates, are not redistributed.
* **`kampala.csv`** is the analysis-ready CSV of the openwashdata package, unchanged. `datasets.load_kampala()` groups its
  ten categories into six fractions (food + garden + wood; paper; polythene + plastics; metals; glass; textiles + other).

## Populations used in the article

| Population | Households | Selection |
|---|---|---|
| Solomon Islands | 174 | IDs `AH*`, `AR*`, `R*` with two days of waste (Guadalcanal 80; Malaita: Arabella 37, Kilusakwalo 33, Ambu 24) |
| Vanuatu | 201 | IDs `H*` (Port Vila), `LH*` (Lelepa), `LV*` (Luganville), `SHR*` (Shefa rural), `IFI*` (Ifira) |
| Belize | 209 | household samples with sorted weights (of 268 audited) in Belize City 50, Belmopan 30, Dangriga 30, Orange Walk 30, Toledo rural 29, Bullet Tree Falls 20, Corozal Town 10, Mahogany Heights 10 |
| Kampala | 103 | all households (Bwaise I 39, Bukoto I 32, Ggaba 32) |
