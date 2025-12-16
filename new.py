print("\n=== EVALUATING MODEL (filtering: Type_0 + features_1 -> Type_1) ===")
predictions, actuals = [], []

for i in range(len(test_data)):
    evidence = {('Type', 0): int(test_data[('Type', 0)].iloc[i])}
    for var in ['range_m', 'length_m', 'RCSinst_dB', 'SNRinst_dB']:
        evidence[(var, 1)] = int(test_data[(var, 1)].iloc[i])

    q = inference.query(variables=[('Type', 1)], evidence=evidence, show_progress=False)
    pred = int(np.argmax(q[('Type', 1)].values))
    true = int(test_data[('Type', 1)].iloc[i])

    predictions.append(pred)
    actuals.append(true)

acc = (np.array(predictions) == np.array(actuals)).mean()
print("Accuracy:", acc)