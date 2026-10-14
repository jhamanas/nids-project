# CICIDS2017 — Download Instructions

Unlike NSL-KDD, CICIDS2017 is **not freely scriptable** — the Canadian Institute
for Cybersecurity (CIC) requires you to fill in a short access form before you
get a download link, and the host does not allow bulk/automated retrieval. This
has to be done manually, once, in a browser.

## Steps

1. Go to the official dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
2. Click through to the **download request form** (CIC gates the dataset behind
   a short form — name, email, institution/purpose). Submit it.
3. You'll be given access to an FTP/HTTP directory containing:
   - `MachineLearningCSV.zip` — pre-extracted flow-based CSVs (one CSV per day/attack scenario, ~225 MB zipped). **This is the minimum you need for Phase 1 exploration.**
   - `GeneratedLabelledFlows.zip` — same data, alternate packaging
   - `PCAPs/` — the raw packet captures (**required for Phase 2**, payload-level
     feature extraction — this is the whole point of the project, so don't skip
     it, but it is large: tens of GB across all days)
4. Download `MachineLearningCSV.zip` first (fast, gets you unblocked for Phase 1),
   then start the PCAP download in the background — it can take a while depending
   on your connection.

## Where to put the files

Unzip everything into `data/raw/cicids2017/` so it looks like this:

```
data/raw/cicids2017/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── PCAPs/
    ├── Monday-WorkingHours.pcap
    ├── Tuesday-WorkingHours.pcap
    └── ... (one per day)
```

(Exact filenames vary slightly by mirror/zip — what matters is one CSV per
day/scenario, plus a `PCAPs/` subfolder if you downloaded the raw captures.)

## What each day contains (for Phase 1 exploration)

| Day | Traffic |
|---|---|
| Monday | Benign only (baseline "normal" traffic) |
| Tuesday | Benign + Brute Force (FTP-Patator, SSH-Patator) |
| Wednesday | Benign + DoS/DDoS (Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) |
| Thursday (morning) | Benign + Web Attacks (Brute Force, XSS, SQL Injection) |
| Thursday (afternoon) | Benign + Infiltration |
| Friday (morning) | Benign only |
| Friday (afternoon) | Benign + PortScan + Botnet + DDoS |

This is why **Monday has no attacks** — it exists purely to build your "normal
traffic" profile, which matters a lot for the PAYL baseline in Phase 4.

## Verifying your download

Once files are in place, run:

```bash
python scripts/verify_cicids2017.py
```

This checks that the expected CSVs are present and readable, and prints row
counts + label distribution per file, without needing you to open each one
manually.

## A note on disk space

Full CSVs: ~1.5 GB combined. Full PCAPs: ~50+ GB combined. If you're tight on
space, you can do Phase 1 exploration with just the CSVs and only pull the PCAP
for one or two days to develop the Phase 2 parsing code against, before
committing to downloading all of them.
