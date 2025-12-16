import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import DBN
from pgmpy.inference import DBNInference


# =========================
# 0) EXPECTED INPUT COLUMNS
# =========================
# df must have:
# time_s, Type, range_m, length_m, RCSinst_dB, SNRinst_dB, mc_id


# =========================
# 1) CONFIG
# =========================
TIME_SLICE_SIZE = 100          # seconds per slice (you chose 100)
N_BINS = 5                     # discretization bins for continuous vars
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8

VARS = ["Type", "range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]
CONTINUOUS_VARS = ["range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]


# =========================
# 2) BASIC CLEANING + TIME SLICES
# =========================
df = df.copy()

# Keep only needed columns (avoid accidental extra columns)
df = df[["time_s", "Type", "range_m", "length_m", "RCSinst_dB", "SNRinst_dB", "mc_id"]]

# Drop rows with missing required values
df = df.dropna(subset=["time_s", "Type", "range_m", "length_m", "RCSinst_dB", "SNRinst_dB", "mc_id"])

# Ensure integer Type (4 classes) and mc_id present
df["Type"] = df["Type"].astype(int)

# Create time slices
df["time_slice"] = (df["time_s"] // TIME_SLICE_SIZE).astype(int)

# Sort for deterministic sequence processing
df = df.sort_values(["mc_id", "time_slice", "time_s"]).reset_index(drop=True)


# =========================
# 3) ANSWER YOUR "LAST ONE" QUESTION:
#    Do we have multiple rows per (mc_id, time_slice)?
# =========================
# This tells you if a time slice contains multiple samples.
counts = df.groupby(["mc_id", "time_slice"]).size()
multi_ratio = (counts > 1).mean()  # fraction of (mc_id,time_slice) pairs with >1 row
max_per_slice = counts.max()

print("=== (mc_id, time_slice) multiplicity check ===")
print(f"Total (mc_id,time_slice) groups: {len(counts):,}")
print(f"Fraction of groups with >1 row: {multi_ratio:.3f}")
print(f"Max rows in a single (mc_id,time_slice): {int(max_per_slice)}")
print()


# =========================
# 4) DISCRETIZE CONTINUOUS VARS
#    Use quantile to keep bins balanced (usually better than uniform)
# =========================
disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")

# KBinsDiscretizer returns floats; we convert to int
df[CONTINUOUS_VARS] = disc.fit_transform(df[CONTINUOUS_VARS]).astype(int)

# (Optional) sanity: ensure bins are within [0, N_BINS-1]
for c in CONTINUOUS_VARS:
    if df[c].min() < 0 or df[c].max() >= N_BINS:
        raise ValueError(f"Discretization out of expected range for {c}: {df[c].min()}..{df[c].max()}")


# =========================
# 5) BUILD TRANSITION DATAFRAME
#    Each row is one transition: slice t -> slice t+1 within each mc_id
#
# IMPORTANT: since you likely have multiple samples per time_slice,
# we must aggregate within each (mc_id,time_slice).
#
# Here we use "first" sample per slice.
# If you prefer: median/mode/etc, tell me and I’ll swap it.
# =========================
rows = []
bad_sequences = 0

for seq_id, g in df.groupby("mc_id", sort=False):
    # aggregate to exactly 1 row per time_slice for this sequence
    # using first sample in each slice
    g_slice = g.groupby("time_slice", as_index=False)[VARS].first()

    # need at least 2 slices to form 1 transition
    if len(g_slice) < 2:
        bad_sequences += 1
        continue

    # create transition pairs
    for t in range(len(g_slice) - 1):
        r = {"mc_id": seq_id}  # keep for splitting only (NOT part of DBN)
        for v in VARS:
            r[(v, 0)] = int(g_slice.loc[t, v])
            r[(v, 1)] = int(g_slice.loc[t + 1, v])
        rows.append(r)

dbn_df = pd.DataFrame(rows)

print("=== Transition dataset ===")
print(f"Transitions (rows): {len(dbn_df):,}")
print(f"Sequences skipped (too short): {bad_sequences:,}")
print("Columns:", len(dbn_df.columns))
print()


# =========================
# 6) SPLIT TRAIN/TEST BY mc_id (NOT by global slice number)
# =========================
rng = np.random.default_rng(RANDOM_SEED)
all_ids = dbn_df["mc_id"].unique()
rng.shuffle(all_ids)

cut = int(TRAIN_FRACTION * len(all_ids))
train_ids = set(all_ids[:cut])
test_ids = set(all_ids[cut:])

train_data = dbn_df[dbn_df["mc_id"].isin(train_ids)].drop(columns=["mc_id"]).reset_index(drop=True)
test_data  = dbn_df[dbn_df["mc_id"].isin(test_ids)].drop(columns=["mc_id"]).reset_index(drop=True)

print("=== Split ===")
print(f"Unique mc_id total: {len(all_ids):,}")
print(f"Train mc_id: {len(train_ids):,} | Test mc_id: {len(test_ids):,}")
print(f"Train transitions: {len(train_data):,} | Test transitions: {len(test_data):,}")
print()


# =========================
# 7) DEFINE DBN STRUCTURE (NO mc_id node!)
# =========================
model = DBN([
    # Intra-slice (template) edges: features -> Type
    (('range_m', 0), ('Type', 0)),
    (('length_m', 0), ('Type', 0)),
    (('RCSinst_dB', 0), ('Type', 0)),
    (('SNRinst_dB', 0), ('Type', 0)),

    # Inter-slice edges (temporal)
    (('Type', 0), ('Type', 1)),
    (('range_m', 0), ('range_m', 1)),
    (('length_m', 0), ('length_m', 1)),
    (('RCSinst_dB', 0), ('RCSinst_dB', 1)),
    (('SNRinst_dB', 0), ('SNRinst_dB', 1)),
])


# =========================
# 8) FIT (MLE)
# =========================
print("=== FITTING DBN ===")
model.fit(train_data, estimator="MLE")
model.check_model()

print(f"✓ Fit complete. CPDs: {len(model.cpds)}")
print("Cardinalities:", model.get_cardinality())
print()


# =========================
# 9) INFERENCE (should NOT OOM now)
# =========================
print("=== BUILDING INFERENCE ENGINE ===")
inference = DBNInference(model)
print("✓ DBNInference ready")
print()


# =========================
# 10) EVALUATION
# Two modes:
#   A) FILTERING MODE (easier): use Type_0 + features_1 to predict Type_1
#   B) PURE CLASSIFICATION (harder): use features_1 only to predict Type_1
# =========================
def eval_dbn(test_data: pd.DataFrame, use_prev_type: bool) -> float:
    preds = []
    trues = []

    for i in range(len(test_data)):
        ev = {
            ('range_m', 1): int(test_data.iloc[i][('range_m', 1)]),
            ('length_m', 1): int(test_data.iloc[i][('length_m', 1)]),
            ('RCSinst_dB', 1): int(test_data.iloc[i][('RCSinst_dB', 1)]),
            ('SNRinst_dB', 1): int(test_data.iloc[i][('SNRinst_dB', 1)]),
        }
        if use_prev_type:
            ev[('Type', 0)] = int(test_data.iloc[i][('Type', 0)])

        q = inference.query(variables=[('Type', 1)], evidence=ev, show_progress=False)
        pred = int(np.argmax(q[('Type', 1)].values))
        true = int(test_data.iloc[i][('Type', 1)])

        preds.append(pred)
        trues.append(true)

    preds = np.array(preds, dtype=int)
    trues = np.array(trues, dtype=int)
    return float((preds == trues).mean())


print("=== EVALUATION ===")
acc_filtering = eval_dbn(test_data, use_prev_type=True)
acc_pure      = eval_dbn(test_data, use_prev_type=False)

print(f"Accuracy (Filtering: uses Type_0 + features_1 -> Type_1): {acc_filtering:.4f}")
print(f"Accuracy (Pure: features_1 -> Type_1):               {acc_pure:.4f}")