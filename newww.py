import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from pgmpy.models import DBN
from pgmpy.inference import DBNInference


TIME_SLICE_SIZE = 100   # seconds; you can try 50 or 20 if you want more slices
N_BINS = 5
RANDOM_SEED = 42

VARS = ["Type", "range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]
CONT = ["range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]


df = df.copy()

df["Type"] = df["Type"].astype(int)
df["mc_id"] = df["mc_id"].astype(int)
df["time_s"] = df["time_s"].astype(float)

df["time_slice"] = (df["time_s"] // TIME_SLICE_SIZE).astype(int)

print("Unique mc_id:", df["mc_id"].nunique())
print("Unique Type:", df["Type"].nunique())
print("Unique time slices overall:", df["time_slice"].nunique())
print(df[["mc_id","time_s","time_slice","Type"]].head())



def mode_safe(x):
    m = x.mode()
    return int(m.iloc[0]) if len(m) else int(x.iloc[0])

agg = (
    df.groupby(["mc_id", "time_slice"], as_index=False)
      .agg({
          "Type": mode_safe,
          "range_m": "median",
          "length_m": "median",
          "RCSinst_dB": "mean",
          "SNRinst_dB": "mean",
      })
)

print("agg shape:", agg.shape)
print("Unique time slices in agg:", agg["time_slice"].nunique())
print(agg.head())



le = LabelEncoder()
agg["Type"] = le.fit_transform(agg["Type"].astype(int)).astype(int)

print("Type mapping (original -> encoded):")
print(dict(zip(le.classes_, le.transform(le.classes_))))

disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")
agg[CONT] = disc.fit_transform(agg[CONT]).astype(int)

# quick sanity
print("Encoded Type unique:", sorted(agg["Type"].unique()))
print("Any NaN in agg?", agg.isna().any().any())





rows = []
for mc, g in agg.sort_values(["mc_id", "time_slice"]).groupby("mc_id"):
    g = g.reset_index(drop=True)
    ts = g["time_slice"].to_numpy()

    for i in range(len(g) - 1):
        if ts[i+1] != ts[i] + 1:
            continue  # skip gaps

        r = {"mc_id": int(mc), "t0": int(ts[i]), "t1": int(ts[i+1])}
        for v in VARS:
            r[(v, 0)] = int(g.loc[i, v])
            r[(v, 1)] = int(g.loc[i+1, v])
        rows.append(r)

dbn_df = pd.DataFrame(rows)

# make DBN columns MultiIndex; keep mc_id, t0, t1 as plain columns
meta_cols = ["mc_id", "t0", "t1"]
mi_cols = [c for c in dbn_df.columns if isinstance(c, tuple)]
dbn_df[mi_cols] = dbn_df[mi_cols].astype(int)
dbn_df.columns = [c if c in meta_cols else c for c in dbn_df.columns]
dbn_df.columns = pd.Index(dbn_df.columns)  # clean

# convert only tuple columns to MultiIndex
dbn_df_mi = dbn_df.copy()
dbn_df_mi.columns = [c if isinstance(c, tuple) else (c, "") for c in dbn_df_mi.columns]
dbn_df_mi.columns = pd.MultiIndex.from_tuples(dbn_df_mi.columns)

print("dbn transitions shape:", dbn_df_mi.shape)
print("Unique mc_id in transitions:", dbn_df["mc_id"].nunique())
print("Transitions per mc_id (describe):")
print(dbn_df.groupby("mc_id").size().describe())
print("Unique (t0,t1) pairs:", dbn_df[["t0","t1"]].drop_duplicates().shape[0])




ids = dbn_df["mc_id"].unique()
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(ids)

cut = int(0.8 * len(ids))
train_ids = set(ids[:cut])
test_ids = set(ids[cut:])

train_df = dbn_df[dbn_df["mc_id"].isin(train_ids)].copy()
test_df  = dbn_df[dbn_df["mc_id"].isin(test_ids)].copy()

# Drop meta columns for fitting/inference
train_data = train_df.drop(columns=["mc_id","t0","t1"])
test_data  = test_df.drop(columns=["mc_id","t0","t1"])

# Convert to MultiIndex columns expected by pgmpy
train_data.columns = pd.MultiIndex.from_tuples([(c, "") if not isinstance(c, tuple) else c for c in train_data.columns])
test_data.columns  = pd.MultiIndex.from_tuples([(c, "") if not isinstance(c, tuple) else c for c in test_data.columns])

print("train transitions:", train_data.shape, "test transitions:", test_data.shape)



model = DBN([
    # Intra-slice: features -> Type
    (('range_m', 0), ('Types
    (('length_m', 0), ('Type', 0)),
    (('RCSinst_dB', 0), ('Type', 0)),
    (('SNRinst_dB', 0), ('Type', 0)),

    (('range_m', 1), ('Type', 1)),
    (('length_m', 1), ('Type', 1)),
    (('RCSinst_dB', 1), ('Type', 1)),
    (('SNRinst_dB', 1), ('Type', 1)),

    # Inter-slice: temporal persistence
    (('Type', 0), ('Type', 1)),
    (('range_m', 0), ('range_m', 1)),
    (('length_m', 0), ('length_m', 1)),
    (('RCSinst_dB', 0), ('RCSinst_dB', 1)),
    (('SNRinst_dB', 0), ('SNRinst_dB', 1)),
])

model.fit(train_data, estimator="MLE")
model.check_model()
print("CPDs:", len(model.get_cpds()))




inference = DBNInference(model)

def decode_type_posterior(q):
    phi = q[('Type', 1)]
    probs = phi.values.astype(float)
    states = phi.state_names[('Type', 1)]
    return states, probs

def eval_option_C(test_data):
    preds, trues = [], []
    for _, row in test_data.iterrows():
        evidence = {('Type', 0): int(row[('Type', 0)])}
        for v in CONT:
            evidence[(v, 1)] = int(row[(v, 1)])

        q = inference.query(variables=[('Type', 1)], evidence=evidence)
        states, probs = decode_type_posterior(q)
        pred = int(states[int(np.argmax(probs))])
        true = int(row[('Type', 1)])
        preds.append(pred); trues.append(true)

    preds = np.array(preds); trues = np.array(trues)
    return (preds == trues).mean()

def eval_persistence_baseline(test_data):
    # predict Type_1 = Type_0
    preds = test_data[('Type', 0)].astype(int).to_numpy()
    trues = test_data[('Type', 1)].astype(int).to_numpy()
    return (preds == trues).mean()

acc_C = eval_option_C(test_data)
acc_base = eval_persistence_baseline(test_data)

print("Baseline (Type_t = Type_{t-1}):", acc_base)
print("DBN Option C (Type_{t-1} + features_t -> Type_t):", acc_C)



# Pick a test mc_id that has multiple transitions if possible
mc_counts = test_df.groupby("mc_id").size().sort_values(ascending=False)
mc_id = int(mc_counts.index[0])
print("Chosen mc_id:", mc_id, "transitions:", int(mc_counts.iloc[0]))

track = test_df[test_df["mc_id"] == mc_id].sort_values(["t0","t1"]).reset_index(drop=True)

def decode_label(enc):
    # Convert encoded type back to original label
    return int(le.inverse_transform([int(enc)])[0])

for i in range(len(track)):
    t0 = int(track.loc[i, "t0"])
    t1 = int(track.loc[i, "t1"])

    evidence = {('Type', 0): int(track.loc[i, ('Type', 0)])}
    for v in CONT:
        evidence[(v, 1)] = int(track.loc[i, (v, 1)])

    q = inference.query(variables=[('Type', 1)], evidence=evidence)
    states, probs = decode_type_posterior(q)

    pred_enc = int(states[int(np.argmax(probs))])
    pred_prob = float(np.max(probs))
    true_enc = int(track.loc[i, ('Type', 1)])

    # pretty print distribution (top 3)
    pairs = sorted([(int(s), float(p)) for s, p in zip(states, probs)], key=lambda x: x[1], reverse=True)[:3]
    pairs_pretty = [(decode_label(s), round(p, 3)) for s, p in pairs]

    print(f"\nTransition {i+1}: slice {t0} -> {t1}")
    print(f"  True Type(t0): {decode_label(int(track.loc[i, ('Type',0)]))}")
    print(f"  True Type(t1): {decode_label(true_enc)}")
    print(f"  Pred Type(t1): {decode_label(pred_enc)}  (confidence={pred_prob:.3f})")
    print(f"  Posterior top-3: {pairs_pretty}")
    
    
    
    
    
# per-track last slice and number of slices
track_stats = agg.groupby("mc_id")["time_slice"].agg(["min","max","nunique"]).reset_index()
print(track_stats[["nunique","max"]].describe())

print("\nHow many tracks reach slice >= k?")
for k in range(0, 10):
    print(k, (track_stats["max"] >= k).sum())