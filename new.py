import pandas as pd
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

from IPython.display import display

# ===== CONFIG (edit these if needed) =====
CSV_PATH = "radar_tracks.csv"   # <-- change this to your file path

TRACK_COL = "mc_id"             # track / Monte-Carlo run id column
TIME_COL = "time_s"             # time column
CLASS_COL = "Type"              # encoded object type (0..3)

# continuous radar measurements
CONT_COLS = ["range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]

# number of time steps per track to use
T = 20

# number of bins for discretization
N_BINS = 5

# If you want to limit number of tracks per class for speed (optional)
MAX_TRACKS_PER_CLASS = None  # e.g. 200, or leave as None to use all

print("Config:")
print("CSV_PATH:", CSV_PATH)
print("TRACK_COL:", TRACK_COL)
print("TIME_COL:", TIME_COL)
print("CLASS_COL:", CLASS_COL)
print("CONT_COLS:", CONT_COLS)
print("T (time steps per track):", T)
print("N_BINS (for discretization):", N_BINS)
print("MAX_TRACKS_PER_CLASS:", MAX_TRACKS_PER_CLASS)







# Load the CSV
df = pd.read_csv(CSV_PATH)

print("Raw dataframe loaded.")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())

print("\nDtypes:")
print(df.dtypes)

# Check required columns
required = {TRACK_COL, TIME_COL, CLASS_COL} | set(CONT_COLS)
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Sort by track + time to be safe
df = df.sort_values([TRACK_COL, TIME_COL]).reset_index(drop=True)

print("\nAfter sorting by track and time:")
print("Shape:", df.shape)

# Basic checks on label and tracks
print("\nNumber of unique tracks:", df[TRACK_COL].nunique())
print("Track id sample:", df[TRACK_COL].unique()[:10])

print("\nClass distribution (Type):")
print(df[CLASS_COL].value_counts())







# One label per track (assume Type is constant within a track)
track_labels = (
    df.groupby(TRACK_COL)[CLASS_COL]
    .agg(lambda x: x.iloc[0])
    .to_frame("label")
    .reset_index()
)

print("Track labels (one row per track):")
display(track_labels.head())

print("\nClass distribution over tracks:")
print(track_labels["label"].value_counts())

# Train/test split by track and by class
test_ratio = 0.3
rng = np.random.RandomState(42)

train_track_ids = []
test_track_ids = []

for cls, group in track_labels.groupby("label"):
    track_ids_cls = group[TRACK_COL].tolist()
    rng.shuffle(track_ids_cls)
    
    if MAX_TRACKS_PER_CLASS is not None:
        track_ids_cls = track_ids_cls[:MAX_TRACKS_PER_CLASS]
    
    n_total = len(track_ids_cls)
    n_test = max(1, int(test_ratio * n_total))
    
    test_ids_cls = track_ids_cls[:n_test]
    train_ids_cls = track_ids_cls[n_test:]
    
    train_track_ids.extend(train_ids_cls)
    test_track_ids.extend(test_ids_cls)
    
    print(f"\nClass {cls}: total tracks used = {n_total}")
    print(f"  Train: {len(train_ids_cls)}, Test: {len(test_ids_cls)}")

train_track_ids = np.array(train_track_ids)
test_track_ids = np.array(test_track_ids)

print("\n=== Overall split ===")
print("Total train tracks:", len(train_track_ids))
print("Total test tracks:", len(test_track_ids))
print("Sample train ids:", train_track_ids[:10])
print("Sample test ids:", test_track_ids[:10])

# Sub-dataframes
df_train = df[df[TRACK_COL].isin(train_track_ids)].copy()
df_test  = df[df[TRACK_COL].isin(test_track_ids)].copy()

print("\nTrain dataframe shape:", df_train.shape)
print("Test dataframe shape:", df_test.shape)





print("Basic stats for continuous columns (TRAIN only):")
display(df_train[CONT_COLS].describe())

# Optional: peek at value ranges
for col in CONT_COLS:
    print(f"\nColumn: {col}")
    print("  Min:", df_train[col].min())
    print("  Max:", df_train[col].max())
    
    
    
    
# Fit KBinsDiscretizer on TRAIN continuous columns
disc = KBinsDiscretizer(
    n_bins=N_BINS,
    encode="ordinal",
    strategy="quantile"
)

disc.fit(df_train[CONT_COLS])

print("KBinsDiscretizer fitted.")
print("\nBin edges per feature (approx quantiles):")
for col, edges in zip(CONT_COLS, disc.bin_edges_):
    print(f"{col}: {edges}")

# Apply discretizer to ALL data
df_disc = df.copy()
df_disc[CONT_COLS] = disc.transform(df_disc[CONT_COLS]).astype(int)

print("\nAfter discretization: head of discretized df:")
display(df_disc.head())

# Check unique bins used for each feature
for col in CONT_COLS:
    print(f"\nDiscretized {col} value counts:")
    print(df_disc[col].value_counts().sort_index())
    
    
    
    
# We will create a wide table:
# columns like (Type, 0), (range_m, 0), ..., (Type, 1), (range_m, 1), ...

rows = []
used_train_ids = []

for tid in train_track_ids:
    g = df_disc[df_disc[TRACK_COL] == tid].sort_values(TIME_COL)
    n_steps = len(g)
    
    if n_steps < T:
        # Skip tracks that are too short
        continue
    
    g = g.iloc[:T].reset_index(drop=True)
    
    row = {}
    for t in range(T):
        # class variable at time t
        row[(CLASS_COL, t)] = int(g.loc[t, CLASS_COL])
        # discretized continuous features
        for col in CONT_COLS:
            row[(col, t)] = int(g.loc[t, col])
    
    rows.append(row)
    used_train_ids.append(tid)

dbn_data_train = pd.DataFrame(rows)

print("DBN training DataFrame built.")
print("Shape:", dbn_data_train.shape)
print("Number of tracks actually used (len >= T):", len(used_train_ids))

print("\nColumns (first 20):")
print(dbn_data_train.columns[:20])

print("\nHead of dbn_data_train:")
display(dbn_data_train.head())

# Check that Type over time is constant within a row (sanity check)
print("\nCheck class consistency for first few rows:")
for i in range(min(3, len(dbn_data_train))):
    row_types = [dbn_data_train[(CLASS_COL, t)].iloc[i] for t in range(T)]
    print(f"Row {i} Type sequence:", row_types)
    
    
    
    
# Build DBN structure
dbn_model = DBN()

# Intra-slice edges (at time 0)
intra_edges = []
for col in CONT_COLS:
    intra_edges.append(((CLASS_COL, 0), (col, 0)))

# Temporal edges: slice 0 -> slice 1
inter_edges = [((CLASS_COL, 0), (CLASS_COL, 1))]
for col in CONT_COLS:
    inter_edges.append(((col, 0), (col, 1)))

print("Intra-slice edges:")
for e in intra_edges:
    print(" ", e)

print("\nTemporal edges:")
for e in inter_edges:
    print(" ", e)

dbn_model.add_edges_from(intra_edges + inter_edges)

print("\nNodes in the model:")
print(dbn_model.nodes())

print("\nEdges in the model:")
print(dbn_model.edges())

# Train / fit CPDs from dbn_data_train
print("\nFitting DBN model on training data...")
dbn_model.fit(dbn_data_train)

print("Done fitting. Checking model...")
dbn_model.check_model()
print("Model check passed ✅")

# Initialize initial state for DBNInference
dbn_model.initialize_initial_state()
print("Initial state initialized for inference.")




print("CPD for Type at time slice 0 (prior):")
cpd_type0 = dbn_model.get_cpds((CLASS_COL, 0))
print(cpd_type0)

# Example: CPD of one feature at t=0 given Type_0
feat = CONT_COLS[2]  # RCSinst_dB, for example
print(f"\nCPD for {feat} at t=0 given Type_0:")
cpd_feat0 = dbn_model.get_cpds((feat, 0))
print(cpd_feat0)




# We'll write the classification logic inline (no function),
# but you can copy/paste as needed.

dbn_inf = DBNInference(dbn_model)
print("DBNInference object created.")


# Let's test on ONE example track first to see the evidence dict
example_tid = test_track_ids[0]
print("\nUsing example test track id:", example_tid)

g_example = df_disc[df_disc[TRACK_COL] == example_tid].sort_values(TIME_COL).reset_index(drop=True)
print("Example track length:", len(g_example))

# Use up to T steps (or fewer if shorter)
T_eff_example = min(T, len(g_example))
g_example = g_example.iloc[:T_eff_example]

print("\nExample track (first few rows after discretization):")
display(g_example[[TRACK_COL, TIME_COL, CLASS_COL] + CONT_COLS].head())

# Build evidence dictionary for this track
evidence_example = {}
for t in range(T_eff_example):
    for col in CONT_COLS:
        evidence_example[(col, t)] = int(g_example.loc[t, col])

print("\nEvidence dictionary keys (first 20):")
print(list(evidence_example.keys())[:20])

print("\nRunning forward inference on example track...")
last_t = T_eff_example - 1
result_example = dbn_inf.forward_inference(
    variables=[(CLASS_COL, last_t)],
    evidence=evidence_example
)

dist_example = result_example[(CLASS_COL, last_t)]
print("\nPosterior over Type at final time step:")
print("Values:", dist_example.values)
print("State names:", dist_example.state_names)

pred_class_example = int(dist_example.values.argmax())
true_class_example = int(g_example[CLASS_COL].iloc[0])

print(f"\nTrue class: {true_class_example}")
print(f"Predicted class: {pred_class_example}")






y_true = []
y_pred = []

print("Running classification on all test tracks...\n")

for idx, tid in enumerate(test_track_ids):
    g = df_disc[df_disc[TRACK_COL] == tid].sort_values(TIME_COL).reset_index(drop=True)
    if g.empty:
        continue
    
    true_label = int(g[CLASS_COL].iloc[0])
    
    T_eff = min(T, len(g))
    g_slice = g.iloc[:T_eff]
    
    # Build evidence
    evidence = {}
    for t in range(T_eff):
        for col in CONT_COLS:
            evidence[(col, t)] = int(g_slice.loc[t, col])
    
    # Inference
    last_t = T_eff - 1
    result = dbn_inf.forward_inference(
        variables=[(CLASS_COL, last_t)],
        evidence=evidence
    )
    
    dist = result[(CLASS_COL, last_t)]
    probs = dist.values
    pred_label = int(probs.argmax())
    
    y_true.append(true_label)
    y_pred.append(pred_label)
    
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(test_track_ids)} test tracks...")

print("\nDone classifying all test tracks.")

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nNumber of evaluated tracks:", len(y_true))

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n=== RESULTS ===")
print(f"Accuracy: {acc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm)

# Classification report
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))




