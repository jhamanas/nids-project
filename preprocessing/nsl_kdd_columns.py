"""
NSL-KDD column names and attack-category mapping.

The raw KDDTrain+.txt / KDDTest+.txt files ship with NO header row, so we define
the 43 columns (41 features + label + difficulty_level) here once and import
them everywhere else, rather than repeating this list in every script.

Reference: Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set," 2009.
"""

# 41 KDD features + the label (specific attack name, or "normal") + difficulty_level
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

# Maps each specific attack label found in the dataset to its broad category.
# Anything not in this dict (including "normal") is handled separately by the
# loader — see preprocessing/nsl_kdd_loader.py in Phase 2/3.
ATTACK_CATEGORY_MAP = {
    # --- DoS ---
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # --- Probe ---
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # --- R2L (Remote to Local) ---
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # --- U2R (User to Root) ---
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def label_to_category(label: str) -> str:
    """Map a raw NSL-KDD label to 'Normal' or its broad attack category."""
    label = label.strip().lower()
    if label == "normal":
        return "Normal"
    return ATTACK_CATEGORY_MAP.get(label, "Unknown")
