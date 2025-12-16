import numpy as np
from pgmpy.inference import DBNInference

inference = DBNInference(model)

predictions, actuals = [], []
need_cols = [
    ('Type', 0), ('Type', 1),
    ('range_m', 1), ('length_m', 1),
    ('RCSinst_dB', 1), ('SNRinst_dB', 1)
]

skipped = 0
for i in range(len(test_data)):
    row = test_data.iloc[i]

    # safety check
    if row[need_cols].isna().any():
        skipped += 1
        continue

    evidence = {('Type', 0): int(row[('Type', 0)])}
    for var in ['range_m', 'length_m', 'RCSinst_dB', 'SNRinst_dB']:
        evidence[(var, 1)] = int(row[(var, 1)])

    q = inference.query(
        variables=[('Type', 1)],
        evidence=evidence
    )

    pred = int(np.argmax(q[('Type', 1)].values))
    true = int(row[('Type', 1)])

    predictions.append(pred)
    actuals.append(true)

acc = (np.array(predictions) == np.array(actuals)).mean()
print(f"Accuracy (Option C): {acc:.4f}")
print(f"Used rows: {len(predictions)} | Skipped rows: {skipped}")