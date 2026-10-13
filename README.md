# Inner-Rank Ranked Set Sampling — replication code

Replication code for

> J. Almutawa (2026). *Inner-Rank Ranked Set Sampling: A Stratified
> Alternative to the Concentration Step.*

Every table, figure and in-text numerical check in the paper is produced by a
script in this repository. Nothing is hand-computed.

## Quick start

```
git clone https://github.com/<your-username>/inner-rank-rss.git
cd inner-rank-rss
python3.10 -m pip install numpy scipy scikit-learn matplotlib pandas
python3.10 -u exp18_crossover.py | tee exp18.txt
```

## Reproducibility

Every script fixes its own seeds and is deterministic within a computing
environment. The only entries that vary between environments are those
produced by `scikit-learn`'s Fast-MCD, whose random subset search depends on
library version and numerical backend; Section 5.1 of the paper quantifies the
variation table by table and no conclusion depends on it. The multistart MCD
search used for the decisive comparison in Section 5.8 is implemented here
rather than called from a library, and is therefore exactly reproducible
across environments.

Reference-environment output (AMD Ryzen 7 5700U, Python 3.10) is in
`logs_reference/`; a second environment is in `logs_second_env/`. Run
`python3.10 verify_logs.py` to compare the two numerically.

Manuscript: *Inner-Rank Ranked Set Sampling: A Stratified Alternative to the
Concentration Step.*

Every table, figure and in-text numerical check in the paper is produced by one
of the scripts below. All scripts are self-contained, fix their own seeds, and
print their results to standard output. Requirements: Python 3.10+, `numpy`,
`scipy`, `scikit-learn`, `matplotlib` (figures only), `pandas` (air-quality
analysis only).

Run any script as, for example:

    python3.10 -u exp18_crossover.py | tee exp18.txt

## Scripts and what they produce

| Script | Produces |
|---|---|
| `exp01_prop1.py` | Table 2 (balanced-design null), delta-method standard errors |
| `exp02_prop2.py` | Proposition 2 order-statistic checks |
| `exp03_mechanism.py` | mechanism diagnostics |
| `exp04_window_se.py` | window standard errors |
| `exp05_allocation.py` | Tables 3 and 4 (allocation, contamination) |
| `exp06_pilot.py` | pilot sensitivity |
| `exp07_poolpilot.py` | Table 9 (realisable pilots) |
| `exp08_theory_check.py` | Theorems 4 and 6 and the bias-transmission law |
| `exp09_nonelliptical.py` | Tables 12 and 13 (heavy tails, skewness) |
| `exp10_competitors.py` | Table 11 (full-pool competitors, timings) |
| `exp11_scatter_alloc.py` | Table 6 (scatter allocation) |
| `exp12_airquality.py` | Tables 15 and 16 (air quality, Models A and B) |
| `exp13_revision.py` | Clopper–Pearson intervals, overlap bound, Table 14 (scope) |
| `exp13b_checks.py` | known-mean and linearity checks |
| `exp14_se.py` | standard errors for Tables 3, 4, 7 and 13 |
| `exp15_splitequal.py` | equal-size split-pilot comparison (Section 5.6) |
| `exp16_mcdcheck.py` | MCD subset diagnostic and the Sigma_0 = 3 Sigma check |
| `exp17_poolpilot_theory.py` | Table 10 (how much theory survives a pool pilot) |
| `exp18_crossover.py` | Table 12 (paired crossover with bootstrap intervals) |
| `make_figures.py` | Figures 1–3 |
| `verify_logs.py` | checks the archived reference logs against raw `tee` output |

## Logs

`logs_reference/` holds the output of the reference environment (the author's
workstation), which is the environment whose numbers appear in the manuscript.
`logs_second_env/` holds the same scripts run in a second environment; the
differences are confined to `scikit-learn`'s Fast-MCD and are quantified in
Section 5.1 of the paper. Remaining `exp*.log` files in this directory are the
reference-environment outputs of the earlier scripts.

Run `python3.10 verify_logs.py` with the raw `exp15.txt`..`exp18.txt` beside
it to confirm that the archived logs carry the same numbers as the console
output; it ignores the provenance header and the `elapsed` line and compares
every numeric token.

The multistart search in `exp16_mcdcheck.py` and `exp18_crossover.py` is
implemented in this repository rather than called from a library, and is
therefore exactly reproducible across environments from the stated seeds.

## Data

`AirQualityUCI.csv` is the UCI Air Quality data set
(DOI 10.24432/C59K5F), included unmodified for reproducibility.
