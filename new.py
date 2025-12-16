import numpy as np
from pgmpy.inference import DBNInference

inference = DBNInference(model)

# Get the state labels for Type_1 in the correct order
type1_states = model.get_cpds(('Type', 1)).state_names[('Type', 1)]
# Example: ['1','2','4'] or ['0','1','2','3']
print("Type_1 states order:", type1_states)

predictions, actuals = [], []
skipped = 0

need_cols = [
    ('Type', 0), ('Type', 1),
    ('range_m', 1), ('length_m', 1), ('RCSinst_dB', 1), ('SNRinst_dB', 1)
]

for i in range(len(test_data)):
    row = test_data.iloc[i]
    if row[need_cols].isna().any():
        skipped += 1
        continue

    evidence = {
        ('Type', 0): str(int(row[('Type', 0)]))  # match string state names
    }
    for var in ['range_m', 'length_m', 'RCSinst_dB', 'SNRinst_dB']:
        evidence[(var, 1)] = int(row[(var, 1)])  # these should be 0..4 ints

    q = inference.query(variables=[('Type', 1)], evidence=evidence)

    probs = q[('Type', 1)].values
    pred_state = type1_states[int(np.argmax(probs))]   # <-- map index -> label
    true_state = str(int(row[('Type', 1)]))            # ensure same type

    predictions.append(pred_state)
    actuals.append(true_state)

acc = (np.array(predictions) == np.array(actuals)).mean() if predictions else np.nan
print(f"Accuracy (Option C): {acc:.4f}")
print(f"Used rows: {len(predictions)} | Skipped: {skipped}")