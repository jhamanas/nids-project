# CICIDS2017 raw data

This folder holds the raw CICIDS2017 dataset. Nothing here is committed to git
(see `.gitignore`); each person downloads it locally.

Dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
(The direct download links below sit behind the Canadian Institute for
Cybersecurity's download page. Registration may be required.)

## What you need

| Item | Used for | Expected location |
|---|---|---|
| PCAP files (Monday–Friday, `*.pcap`) | Phase 2–3: payload extraction | `data/raw/cicids2017/PCAPs/` |
| `GeneratedLabelledFlows.zip` | Phase 6 / spot-check: flow labels with IP, port, timestamp, Label | `data/raw/cicids2017/GeneratedLabelledFlows/` |
| `MachineLearningCSV.zip` | Flow-feature baseline only (no IP/port/timestamp) | `data/raw/cicids2017/MachineLearningCSV/` |

## Which flow-CSV variant to use

CICIDS2017 ships two CSV variants. They are NOT interchangeable:

- **`GeneratedLabelledFlows.zip` (85 columns)** keeps `Flow ID`, `Source IP`,
  `Source Port`, `Destination IP`, `Destination Port`, `Protocol`, `Timestamp`
  and `Label`. Use this one for anything that needs to match flows to packets
  or payloads.
- **`MachineLearningCSV.zip` (79 columns)** strips the identifying columns on
  purpose, to stop models learning IP addresses. It only has `Destination Port`
  and `Label` as identifiers, so it cannot be used to verify a specific
  anomaly.

Column names in these CSVs carry a leading space (`" Source IP"`, `" Label"`).
`spot_check_labels.py` strips this on load.

## Files used by the spot-check (Friday)

Under `GeneratedLabelledFlows/TrafficLabelling/`:

- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (note the casing: `DDos`)

## Verifying the download

```powershell
python scripts\verify_cicids2017.py
```

To check that a flow CSV has the columns the spot-check needs:

```powershell
python -c "import pandas as pd; df = pd.read_csv(r'data\raw\cicids2017\GeneratedLabelledFlows\TrafficLabelling\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', nrows=5); print([c for c in df.columns])"
```

## Notes

- Run each `Start-BitsTransfer` command on its own line. Pasting two commands
  with no newline between them makes PowerShell concatenate them into one
  invocation, which fails with a `ParameterBindingException`.
- BITS downloads of this dataset ended in `JobState: Error` on at least one
  machine. The manual browser download worked. Check
  `Get-BitsTransfer | Format-List JobId, DisplayName, JobState, ErrorDescription`
  if you want to diagnose it.
- Verify each zip against its `.md5` file before extracting.
