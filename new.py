import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

TIME_SLICE_SIZE = 100
N_BINS = 5
RANDOM_SEED = 42

# 1) time slices
df = df.copy()
df["time_slice"] = (df["time_s"] // TIME_SLICE_SIZE).astype(int)

# 2) aggregate raw samples -> one row per (mc_id, time_slice)
def mode_safe(x):
    m = x.mode()
    return m.iloc[0] if len(m) else x.iloc[0]

agg = (df.groupby(["mc_id", "time_slice"])
         .agg({
             "Type": mode_safe,
             "range_m": "median",
             "length_m": "median",
             "RCSinst_dB": "mean",
             "SNRinst_dB": "mean",
         })
         .reset_index())

# 3) discretize continuous vars on agg
disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")
cont = ["range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]
agg[cont] = disc.fit_transform(agg[cont]).astype(int)

# IMPORTANT: make Type consistent (int labels)
agg["Type"] = agg["Type"].astype(int)



VARS = ["Type", "range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]

rows = []
for mc_id, g in agg.sort_values(["mc_id", "time_slice"]).groupby("mc_id"):
    # g has 1 row per time_slice
    ts = g["time_slice"].to_numpy()
    g = g.reset_index(drop=True)

    # build only consecutive transitions (t -> t+1)
    for i in range(len(g) - 1):
        if ts[i+1] != ts[i] + 1:
            continue  # skip gaps

        r = {"mc_id": mc_id}
        for v in VARS:
            r[(v, 0)] = int(g.loc[i, v])
            r[(v, 1)] = int(g.loc[i+1, v])
        rows.append(r)

dbn_df = pd.DataFrame(rows)
dbn_df.columns = pd.MultiIndex.from_tuples(dbn_df.columns)  # keep mc_id too; we'll drop later
print("dbn_df:", dbn_df.shape)




all_ids = dbn_df[("mc_id",)].unique() if ("mc_id",) in dbn_df.columns else dbn_df["mc_id"].unique()
# easier if mc_id is a normal column:
if "mc_id" in dbn_df.columns:
    ids = dbn_df["mc_id"].unique()
else:
    # if mc_id stored as tuple column ('mc_id',) (rare)
    ids = dbn_df[("mc_id",)].unique()

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(ids)
cut = int(0.8 * len(ids))
train_ids = set(ids[:cut])
test_ids  = set(ids[cut:])

train_data = dbn_df[dbn_df["mc_id"].isin(train_ids)].drop(columns=["mc_id"])
test_data  = dbn_df[dbn_df["mc_id"].isin(test_ids)].drop(columns=["mc_id"])

print("train:", train_data.shape, "test:", test_data.shape)



from pgmpy.models import DBN
from pgmpy.inference import DBNInference

model = DBN([
    # Intra-slice: features -> Type
    (('range_m', 0), ('Type', 0)),
    (('length_m', 0), ('Type', 0)),
    (('RCSinst_dB', 0), ('Type', 0)),
    (('SNRinst_dB', 0), ('Type', 0)),

    # Inter-slice: persistence
    (('Type', 0), ('Type', 1)),
    (('range_m', 0), ('range_m', 1)),
    (('length_m', 0), ('length_m', 1)),
    (('RCSinst_dB', 0), ('RCSinst_dB', 1)),
    (('SNRinst_dB', 0), ('SNRinst_dB', 1)),
])

model.fit(train_data, estimator="MLE")
model.check_model()
print("CPDs:", len(model.cpds))

inference = DBNInference(model)




import numpy as np

def eval_option_B(test_data):
    preds, trues = [], []
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        evidence = {}
        for var in ["range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]:
            evidence[(var, 1)] = int(row[(var, 1)])

        q = inference.query(variables=[('Type', 1)], evidence=evidence)
        type_states = model.get_cpds(('Type', 1)).state_names[('Type', 1)]
        pred = type_states[int(np.argmax(q[('Type', 1)].values))]
        true = str(int(row[('Type', 1)]))  # keep consistent
        preds.append(pred); trues.append(true)
    return (np.array(preds) == np.array(trues)).mean()

acc_B = eval_option_B(test_data)
print("DBN Option B (features_t -> Type_t):", acc_B)



def eval_option_C(test_data):
    preds, trues = [], []
    type_states = model.get_cpds(('Type', 1)).state_names[('Type', 1)]
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        evidence = {('Type', 0): str(int(row[('Type', 0)]))}
        for var in ["range_m", "length_m", "RCSinst_dB", "SNRinst_dB"]:
            evidence[(var, 1)] = int(row[(var, 1)])

        q = inference.query(variables=[('Type', 1)], evidence=evidence)
        pred = type_states[int(np.argmax(q[('Type', 1)].values))]
        true = str(int(row[('Type', 1)]))
        preds.append(pred); trues.append(true)
    return (np.array(preds) == np.array(trues)).mean()

acc_C = eval_option_C(test_data)
print("DBN Option C (Type_{t-1} + features_t -> Type_t):", acc_C)
