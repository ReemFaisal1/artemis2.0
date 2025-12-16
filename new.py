import pandas as pd
import numpy as np
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import DBNInference
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility

np.random.seed(42)

print(”=” * 80)
print(“DYNAMIC BAYESIAN NETWORK FOR RADAR DATA ANALYSIS”)
print(”=” * 80)

# ============================================================================

# STEP 1: LOAD AND PREPROCESS DATA

# ============================================================================

print(”\n[STEP 1] Loading data…”)

# Load your CSV file

# Replace ‘your_radar_data.csv’ with your actual file path

df = pd.read_csv(‘your_radar_data.csv’)

print(f”Data shape: {df.shape}”)
print(f”\nFirst few rows:”)
print(df.head())
print(f”\nData types:”)
print(df.dtypes)
print(f”\nMissing values:”)
print(df.isnull().sum())

# ============================================================================

# STEP 2: DATA PREPARATION FOR DBN

# ============================================================================

print(”\n[STEP 2] Preparing data for DBN…”)

# Sort by mc_id and time to ensure temporal ordering

df = df.sort_values([‘mc_id’, ‘time_s’]).reset_index(drop=True)

# DBN requires discrete variables - discretize continuous variables

print(”\nDiscretizing continuous variables…”)

# Create a copy for discretization

df_discrete = df.copy()

# Discretize continuous features into bins

# You can adjust n_bins based on your data distribution

n_bins = 5  # Using 5 bins for each continuous variable

discretizer_rcs = KBinsDiscretizer(n_bins=n_bins, encode=‘ordinal’, strategy=‘quantile’)
discretizer_snr = KBinsDiscretizer(n_bins=n_bins, encode=‘ordinal’, strategy=‘quantile’)
discretizer_range = KBinsDiscretizer(n_bins=n_bins, encode=‘ordinal’, strategy=‘quantile’)

# Handle ‘length’ - if it’s continuous, discretize it; if categorical, leave as is

if df[‘length’].dtype in [‘float64’, ‘int64’] and df[‘length’].nunique() > 10:
discretizer_length = KBinsDiscretizer(n_bins=n_bins, encode=‘ordinal’, strategy=‘quantile’)
df_discrete[‘length’] = discretizer_length.fit_transform(df[[‘length’]]).astype(int)
else:
df_discrete[‘length’] = df[‘length’].astype(int)

df_discrete[‘rcs’] = discretizer_rcs.fit_transform(df[[‘rcs’]]).astype(int)
df_discrete[‘snr’] = discretizer_snr.fit_transform(df[[‘snr’]]).astype(int)
df_discrete[‘range’] = discretizer_range.fit_transform(df[[‘range’]]).astype(int)

# Encode ‘type’ if it’s categorical

if df[‘type’].dtype == ‘object’:
df_discrete[‘type’] = pd.Categorical(df[‘type’]).codes
else:
df_discrete[‘type’] = df[‘type’].astype(int)

print(”\nDiscretized data sample:”)
print(df_discrete.head(10))

# ============================================================================

# STEP 3: CREATE TIME-SLICED DATA FOR DBN

# ============================================================================

print(”\n[STEP 3] Creating time-sliced data…”)

# DBN requires data in specific format: consecutive time slices

# We’ll create pairs of (t, t+1) observations

def create_time_slices(data, mc_id_col=‘mc_id’, time_col=‘time_s’):
“””
Create time-sliced data for DBN training.
Each row will contain variables at time t and t+1.
“””
sliced_data = []

```
for mc_id in data[mc_id_col].unique():
    mc_data = data[data[mc_id_col] == mc_id].sort_values(time_col)
    
    # Create consecutive pairs
    for i in range(len(mc_data) - 1):
        row_t = mc_data.iloc[i]
        row_t1 = mc_data.iloc[i + 1]
        
        # Create a dictionary with t and t+1 values
        sliced_row = {}
        for col in ['type', 'rcs', 'snr', 'length', 'range']:
            sliced_row[f'{col}_t0'] = row_t[col]
            sliced_row[f'{col}_t1'] = row_t1[col]
        
        sliced_data.append(sliced_row)

return pd.DataFrame(sliced_data)
```

# Create time-sliced dataset

df_sliced = create_time_slices(df_discrete)

print(f”Time-sliced data shape: {df_sliced.shape}”)
print(f”\nTime-sliced data sample:”)
print(df_sliced.head())

# ============================================================================

# STEP 4: SPLIT DATA INTO TRAIN/TEST SETS

# ============================================================================

print(”\n[STEP 4] Splitting data…”)

# Split into train and test sets (80-20 split)

train_data, test_data = train_test_split(df_sliced, test_size=0.2, random_state=42)

print(f”Training set size: {len(train_data)}”)
print(f”Test set size: {len(test_data)}”)

# ============================================================================

# STEP 5: DEFINE DBN STRUCTURE

# ============================================================================

print(”\n[STEP 5] Defining DBN structure…”)

# Define the Dynamic Bayesian Network structure

# Edges are defined as (from_node, to_node, time_relation)

# time_relation=0 means intra-slice (within same time)

# time_relation=1 means inter-slice (from t to t+1)

dbn = DBN()

# Define variables at each time slice

# Format: (variable_name, time_slice)

variables_t0 = [(‘type’, 0), (‘rcs’, 0), (‘snr’, 0), (‘length’, 0), (‘range’, 0)]
variables_t1 = [(‘type’, 1), (‘rcs’, 1), (‘snr’, 1), (‘length’, 1), (‘range’, 1)]

# Add nodes to the DBN

dbn.add_nodes_from([var[0] for var in variables_t0])

# Define edges for the DBN structure

# Inter-slice edges (temporal dependencies from t to t+1)

edges_inter = [
((‘type’, 0), (‘type’, 1)),      # Type at t influences type at t+1
((‘rcs’, 0), (‘rcs’, 1)),        # RCS at t influences RCS at t+1
((‘snr’, 0), (‘snr’, 1)),        # SNR at t influences SNR at t+1
((‘range’, 0), (‘range’, 1)),    # Range at t influences range at t+1
((‘length’, 0), (‘length’, 1)),  # Length at t influences length at t+1

```
# Cross-variable temporal dependencies
(('rcs', 0), ('snr', 1)),        # RCS affects future SNR
(('range', 0), ('snr', 1)),      # Range affects future SNR
(('snr', 0), ('type', 1)),       # SNR affects future type detection
```

]

# Intra-slice edges (dependencies within same time slice at t=0)

edges_intra = [
((‘rcs’, 0), (‘snr’, 0)),        # RCS affects SNR
((‘range’, 0), (‘snr’, 0)),      # Range affects SNR
((‘snr’, 0), (‘type’, 0)),       # SNR affects type detection
]

# Add edges to DBN

dbn.add_edges_from(edges_inter + edges_intra)

print(f”DBN nodes: {dbn.nodes()}”)
print(f”DBN edges: {dbn.edges()}”)

# Visualize the structure (optional)

print(”\nDBN structure defined successfully!”)

# ============================================================================

# STEP 6: PREPARE DATA IN PGMPY FORMAT

# ============================================================================

print(”\n[STEP 6] Preparing data in pgmpy format…”)

# pgmpy expects data with columns named as (variable, timeslice)

# We need to rename our columns to match this format

def format_data_for_pgmpy(data):
“”“Format data for pgmpy DBN.”””
formatted = pd.DataFrame()
for col in [‘type’, ‘rcs’, ‘snr’, ‘length’, ‘range’]:
formatted[(col, 0)] = data[f’{col}_t0’].astype(int)
formatted[(col, 1)] = data[f’{col}_t1’].astype(int)
return formatted

train_formatted = format_data_for_pgmpy(train_data)
test_formatted = format_data_for_pgmpy(test_data)

print(“Formatted training data sample:”)
print(train_formatted.head())

# ============================================================================

# STEP 7: LEARN PARAMETERS FROM DATA

# ============================================================================

print(”\n[STEP 7] Learning DBN parameters…”)

# Fit the DBN model using Maximum Likelihood Estimation

print(“Fitting model with Maximum Likelihood Estimation…”)

try:
dbn.fit(train_formatted, estimator=MaximumLikelihoodEstimator)
print(“✓ Model fitted successfully!”)

```
# Display learned CPDs (Conditional Probability Distributions)
print(f"\nNumber of CPDs learned: {len(dbn.cpds)}")

# Show one example CPD
print("\nExample CPD for 'type' at time slice 1:")
for cpd in dbn.cpds:
    if cpd.variable == 'type' and cpd.variables[0] == ('type', 1):
        print(cpd)
        break
```

except Exception as e:
print(f”✗ Error during fitting: {e}”)
print(”\nTroubleshooting tips:”)
print(“1. Check that all variables are discrete (integer types)”)
print(“2. Verify no missing values in the data”)
print(“3. Ensure sufficient data for each variable state combination”)

# ============================================================================

# STEP 8: INFERENCE AND PREDICTION

# ============================================================================

print(”\n[STEP 8] Making predictions on test set…”)

# Create inference object

dbn_inference = DBNInference(dbn)

# Make predictions on test set

predictions = []
actuals = []

print(“Predicting ‘type’ at t+1 given observations at t…”)

for idx in range(len(test_formatted)):
# Get evidence from time slice 0
evidence = {
(‘type’, 0): int(test_formatted.iloc[idx][(‘type’, 0)]),
(‘rcs’, 0): int(test_formatted.iloc[idx][(‘rcs’, 0)]),
(‘snr’, 0): int(test_formatted.iloc[idx][(‘snr’, 0)]),
(‘length’, 0): int(test_formatted.iloc[idx][(‘length’, 0)]),
(‘range’, 0): int(test_formatted.iloc[idx][(‘range’, 0)])
}

```
# Query the probability distribution for 'type' at t+1
try:
    result = dbn_inference.query([('type', 1)], evidence=evidence)
    
    # Get the most likely state
    probs = result[('type', 1)].values
    predicted_type = np.argmax(probs)
    
    predictions.append(predicted_type)
    actuals.append(int(test_formatted.iloc[idx][('type', 1)]))
    
except Exception as e:
    # If inference fails for this sample, use most common class
    predictions.append(train_formatted[('type', 1)].mode()[0])
    actuals.append(int(test_formatted.iloc[idx][('type', 1)]))

# Progress indicator
if (idx + 1) % 1000 == 0:
    print(f"  Processed {idx + 1}/{len(test_formatted)} samples...")
```

print(f”✓ Predictions complete!”)

# ============================================================================

# STEP 9: EVALUATE MODEL PERFORMANCE

# ============================================================================

print(”\n[STEP 9] Evaluating model performance…”)
print(”=” * 80)

# Calculate accuracy

accuracy = accuracy_score(actuals, predictions)
print(f”\nACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)”)

# Detailed classification report

print(”\nCLASSIFICATION REPORT:”)
print(”-” * 80)
print(classification_report(actuals, predictions))

# Confusion matrix

print(”\nCONFUSION MATRIX:”)
cm = confusion_matrix(actuals, predictions)
print(cm)

# Visualize confusion matrix

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=‘d’, cmap=‘Blues’, cbar=True)
plt.title(‘Confusion Matrix - DBN Predictions’, fontsize=14, fontweight=‘bold’)
plt.ylabel(‘Actual Type’, fontsize=12)
plt.xlabel(‘Predicted Type’, fontsize=12)
plt.tight_layout()
plt.savefig(‘dbn_confusion_matrix.png’, dpi=300, bbox_inches=‘tight’)
print(“✓ Confusion matrix saved as ‘dbn_confusion_matrix.png’”)

# ============================================================================

# STEP 10: MODEL ANALYSIS AND INSIGHTS

# ============================================================================

print(”\n[STEP 10] Model analysis…”)
print(”=” * 80)

# Calculate per-class accuracy

unique_classes = np.unique(actuals)
print(”\nPER-CLASS ACCURACY:”)
for cls in unique_classes:
cls_mask = np.array(actuals) == cls
cls_accuracy = accuracy_score(
np.array(actuals)[cls_mask],
np.array(predictions)[cls_mask]
)
cls_count = np.sum(cls_mask)
print(f”  Class {cls}: {cls_accuracy:.4f} (n={cls_count})”)

# Baseline comparison (predicting most common class)

most_common_class = pd.Series(actuals).mode()[0]
baseline_predictions = [most_common_class] * len(actuals)
baseline_accuracy = accuracy_score(actuals, baseline_predictions)

print(f”\nBASELINE ACCURACY (most common class): {baseline_accuracy:.4f}”)
print(f”IMPROVEMENT OVER BASELINE: {(accuracy - baseline_accuracy)*100:.2f}%”)

# ============================================================================

# FINAL SUMMARY

# ============================================================================

print(”\n” + “=” * 80)
print(“TRAINING COMPLETE - SUMMARY”)
print(”=” * 80)
print(f”Total training samples: {len(train_data):,}”)
print(f”Total test samples: {len(test_data):,}”)
print(f”Number of time slices: 2 (t=0 and t=1)”)
print(f”Number of variables: {len(dbn.nodes())}”)
print(f”Number of edges: {len(dbn.edges())}”)
print(f”\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)”)
print(”=” * 80)

# ============================================================================

# USAGE NOTES

# ============================================================================

print(”\n” + “=” * 80)
print(“USAGE NOTES”)
print(”=” * 80)
print(”””
To use this code with your radar data:

1. Replace ‘your_radar_data.csv’ with your actual file path
1. Adjust discretization bins (n_bins) based on your data distribution
- Use df[‘column’].hist() to visualize distributions
- More bins = more detail but requires more data
1. Modify the DBN structure (edges) based on domain knowledge
- Add edges for variables you believe are related
- Remove edges that don’t make sense for radar physics
1. For better predictions, consider:
- Using more time slices (t-2, t-1, t, t+1, t+2)
- Adding domain-specific features
- Tuning discretization strategies
- Handling class imbalance with SMOTE or class weights
1. For production use:
- Save trained model: dbn.save(‘radar_dbn_model.pkl’)
- Load model: dbn = DBN.load(‘radar_dbn_model.pkl’)
- Implement rolling window predictions for streaming data
1. Advanced techniques:
- Structure learning: use HillClimbSearch or PC algorithm
- Bayesian parameter estimation with priors
- Hidden variables for latent states
- Variable-length sequences with different mc_id runs
  “””)