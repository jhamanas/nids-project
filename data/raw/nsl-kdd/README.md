# NSL-KDD — Dataset Files

Populated automatically by `scripts/download_nsl_kdd.py`. Source mirror:
https://github.com/jmnwong/NSL-KDD-Dataset (the original unb.ca host no longer
serves direct download links; this is a stable, widely-cited mirror of the
same files, originally recovered via the Wayback Machine).

## Files

| File | Rows (approx.) | Purpose |
|---|---|---|
| `KDDTrain+.txt` | 125,973 | Full training set — use for building models |
| `KDDTrain+_20Percent.txt` | 25,192 | 20% subset — use for fast iteration during development |
| `KDDTest+.txt` | 22,544 | Full test set |
| `KDDTest-21.txt` | 11,850 | Harder test subset (excludes records every classifier gets right) |

## Format

Plain CSV, **no header row**. 43 columns: 41 KDD features + `label` (specific
attack name, or `normal`) + `difficulty_level`. Column names and the
attack-name → category (DoS/Probe/R2L/U2R) mapping live in
`preprocessing/nsl_kdd_columns.py` — import from there rather than
hardcoding column names again.

## Role in this project

NSL-KDD is **flow/connection-level**, not raw payload bytes — it's used here as
the flow-based comparison baseline mentioned in the project plan (Phase 6),
not for the PAYL/payload-byte work itself. CICIDS2017 (with its PCAPs) is what
supplies actual payload bytes for Phases 2-5.
