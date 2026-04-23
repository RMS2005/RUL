"""
Run multi_dataset_evaluation.ipynb cells as a script.
Outputs are printed to console and saved to the notebook.
"""
import json
import os
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Change to the notebook directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("  RUNNING MULTI-DATASET EVALUATION")
print("="*60)

# --- Cell 2: Imports ---
# Import TensorFlow FIRST to avoid DLL conflicts with matplotlib on Windows
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# --- Cell 3: Configuration ---
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
RUL_CLIP_VALUE = 125
SEQUENCE_LENGTH = 50
EPOCHS = 50
BATCH_SIZE = 128

IS_KAGGLE = os.path.exists("/kaggle/input")
KAGGLE_DATASET_NAME = "RUL_Project_Dataset"

if IS_KAGGLE:
    DATA_DIR = f"/kaggle/input/{KAGGLE_DATASET_NAME}/C-MPASS/DATASET"
    RESULTS_DIR = "/kaggle/working/results"
else:
    DATA_DIR = "C-MPASS/DATASET"
    RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Cell 5: Data Loading ---
COLUMN_NAMES = (
    ['engine_id', 'time_in_cycles'] +
    [f'setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

def load_dataset(dataset_name):
    train_path = os.path.join(DATA_DIR, f"train_{dataset_name}.txt")
    test_path = os.path.join(DATA_DIR, f"test_{dataset_name}.txt")
    rul_path = os.path.join(DATA_DIR, f"RUL_{dataset_name}.txt")
    
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None)
    truth_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    train_df.drop(columns=[26, 27], inplace=True, errors='ignore')
    test_df.drop(columns=[26, 27], inplace=True, errors='ignore')
    
    train_df.columns = COLUMN_NAMES
    test_df.columns = COLUMN_NAMES
    
    return train_df, test_df, truth_df

# --- Cell 7: Preprocessing ---
def preprocess_dataset(train_df, test_df, truth_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    stats = train_df.describe().transpose()
    constant_cols = stats[stats['std'] == 0].index.tolist()
    low_corr_cols = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16']
    cols_to_drop = list(set(constant_cols + [c for c in low_corr_cols if c in train_df.columns]))
    
    train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    print(f"  Dropped {len(cols_to_drop)} uninformative columns: {cols_to_drop}")
    
    max_cycles = train_df.groupby('engine_id')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycles']
    train_df = pd.merge(train_df, max_cycles, on='engine_id', how='left')
    train_df['RUL'] = train_df['max_cycles'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycles'], inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=RUL_CLIP_VALUE)
    
    feature_cols = train_df.columns.drop(['engine_id', 'time_in_cycles', 'RUL']).tolist()
    
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    print(f"  Features: {len(feature_cols)}, Train samples: {len(train_df)}, Test engines: {test_df['engine_id'].nunique()}")
    
    return train_df, test_df, truth_df, feature_cols, scaler

# --- Cell 9: Sequence Builders ---
def generate_train_sequences(train_df, seq_length, feature_cols):
    sequences, targets = [], []
    for engine_id in train_df['engine_id'].unique():
        engine_df = train_df[train_df['engine_id'] == engine_id]
        for i in range(len(engine_df) - seq_length + 1):
            seq = engine_df[feature_cols].iloc[i:i+seq_length].values
            target = engine_df['RUL'].iloc[i + seq_length - 1]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

def generate_test_sequences(test_df, truth_df, seq_length, feature_cols):
    X_test = []
    for engine_id in test_df['engine_id'].unique():
        engine_df = test_df[test_df['engine_id'] == engine_id]
        last_sequence = engine_df[feature_cols].tail(seq_length).values
        if len(last_sequence) < seq_length:
            padded = np.zeros((seq_length, len(feature_cols)))
            padded[-len(last_sequence):] = last_sequence
            X_test.append(padded)
        else:
            X_test.append(last_sequence)
    return np.array(X_test), truth_df['RUL'].values

# --- Cell 11: Metrics ---
def nasa_s_score(y_true, y_pred):
    d = y_pred.flatten() - y_true.flatten()
    score = 0
    for d_i in d:
        if d_i < 0:
            score += np.exp(-d_i / 13.0) - 1
        else:
            score += np.exp(d_i / 10.0) - 1
    return score

def evaluate_model(y_true, y_pred, model_name, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    s_score = nasa_s_score(y_true, y_pred)
    print(f"    {model_name} | RMSE: {rmse:.2f} | R²: {r2:.4f} | S-Score: {s_score:.2f}")
    return {
        'Dataset': dataset_name, 'Model': model_name,
        'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'NASA S-Score': round(s_score, 2)
    }

# --- Model Builders ---
def build_random_forest():
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

def build_lstm(n_features):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_bilstm(n_features):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
all_results = []
all_predictions = {}

for dataset_name in DATASETS:
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*70}")
    
    train_df, test_df, truth_df = load_dataset(dataset_name)
    print(f"  Raw: train={train_df.shape}, test={test_df.shape}, truth={truth_df.shape}")
    
    train_df, test_df, truth_df, feature_cols, scaler = preprocess_dataset(train_df, test_df, truth_df)
    
    # --- Random Forest ---
    print(f"\n  [1/3] Training Random Forest...")
    X_train_rf = train_df[feature_cols]
    y_train_rf = train_df['RUL']
    X_test_rf = test_df.groupby('engine_id').last()[feature_cols]
    y_test_rf = truth_df['RUL'].values
    
    rf_model = build_random_forest()
    rf_model.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf_model.predict(X_test_rf)
    
    result_rf = evaluate_model(y_test_rf, y_pred_rf, "Random Forest", dataset_name)
    all_results.append(result_rf)
    all_predictions[(dataset_name, "Random Forest")] = (y_test_rf, y_pred_rf)
    
    # --- LSTM ---
    print(f"  [2/3] Training LSTM...")
    X_train_seq, y_train_seq = generate_train_sequences(train_df, SEQUENCE_LENGTH, feature_cols)
    X_test_seq, y_test_seq = generate_test_sequences(test_df, truth_df, SEQUENCE_LENGTH, feature_cols)
    print(f"    Sequences: X_train={X_train_seq.shape}, X_test={X_test_seq.shape}")
    
    lstm_model = build_lstm(n_features=len(feature_cols))
    lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
    result_lstm = evaluate_model(y_test_seq, y_pred_lstm, "LSTM", dataset_name)
    all_results.append(result_lstm)
    all_predictions[(dataset_name, "LSTM")] = (y_test_seq, y_pred_lstm)
    
    # --- Bi-LSTM ---
    print(f"  [3/3] Training Bi-LSTM...")
    bilstm_model = build_bilstm(n_features=len(feature_cols))
    bilstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )
    y_pred_bilstm = bilstm_model.predict(X_test_seq, verbose=0).flatten()
    result_bilstm = evaluate_model(y_test_seq, y_pred_bilstm, "Bi-LSTM", dataset_name)
    all_results.append(result_bilstm)
    all_predictions[(dataset_name, "Bi-LSTM")] = (y_test_seq, y_pred_bilstm)
    
    tf.keras.backend.clear_session()
    print(f"\n  ✓ {dataset_name} complete!")

print(f"\n{'='*70}")
print("  ALL DATASETS EVALUATED SUCCESSFULLY!")
print(f"{'='*70}")

# ============================================================
# RESULTS
# ============================================================
results_df = pd.DataFrame(all_results)
print("\n" + "="*70)
print("  MULTI-DATASET EVALUATION RESULTS")
print("="*70 + "\n")
print(results_df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, "multi_dataset_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to {csv_path}")

# Pivot tables
print("\n─── RMSE Comparison (lower is better) ───")
rmse_pivot = results_df.pivot(index='Dataset', columns='Model', values='RMSE')
rmse_pivot = rmse_pivot[['Random Forest', 'LSTM', 'Bi-LSTM']]
print(rmse_pivot.to_string())

print("\n─── R² Comparison (higher is better) ───")
r2_pivot = results_df.pivot(index='Dataset', columns='Model', values='R²')
r2_pivot = r2_pivot[['Random Forest', 'LSTM', 'Bi-LSTM']]
print(r2_pivot.to_string())

print("\n─── NASA S-Score Comparison (lower is better) ───")
sscore_pivot = results_df.pivot(index='Dataset', columns='Model', values='NASA S-Score')
sscore_pivot = sscore_pivot[['Random Forest', 'LSTM', 'Bi-LSTM']]
print(sscore_pivot.to_string())

# ============================================================
# VISUALIZATIONS
# ============================================================
model_names = ['Random Forest', 'LSTM', 'Bi-LSTM']
colors = ['#2ecc71', '#3498db', '#e74c3c']

# 1. Summary bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics = ['RMSE', 'R²', 'NASA S-Score']
titles = ['RMSE Across Datasets (Lower is Better)', 'R² Across Datasets (Higher is Better)', 'NASA S-Score Across Datasets (Lower is Better)']
for ax, metric, title in zip(axes, metrics, titles):
    pivot = results_df.pivot(index='Dataset', columns='Model', values=metric)
    pivot = pivot[['Random Forest', 'LSTM', 'Bi-LSTM']]
    pivot.plot(kind='bar', ax=ax, rot=0, width=0.7, color=colors)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.legend(title='Model', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "summary_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Summary chart saved to {RESULTS_DIR}/summary_comparison.png")

# 2. Scatter plots
fig, axes = plt.subplots(4, 3, figsize=(18, 22))
for row, dataset_name in enumerate(DATASETS):
    for col, (model_name, color) in enumerate(zip(model_names, colors)):
        ax = axes[row, col]
        y_true, y_pred = all_predictions[(dataset_name, model_name)]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color, edgecolors='none')
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect')
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'RMSE: {rmse:.1f}\nR²: {r2:.3f}', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'{dataset_name} — {model_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('True RUL')
        ax.set_ylabel('Predicted RUL')
        ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs True RUL — All Datasets × All Models', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "scatter_all.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Scatter plots saved to {RESULTS_DIR}/scatter_all.png")

# 3. Error distribution
fig, axes = plt.subplots(4, 3, figsize=(18, 22))
for row, dataset_name in enumerate(DATASETS):
    for col, (model_name, color) in enumerate(zip(model_names, colors)):
        ax = axes[row, col]
        y_true, y_pred = all_predictions[(dataset_name, model_name)]
        errors = y_pred - y_true
        ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        mean_err = np.mean(errors)
        ax.axvline(x=mean_err, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean_err:.1f}')
        over_pct = (errors > 0).sum() / len(errors) * 100
        ax.text(0.95, 0.95, f'Over-est: {over_pct:.0f}%', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'{dataset_name} — {model_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Prediction Error (Pred - True)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
plt.suptitle('Error Distribution — All Datasets × All Models\n(Positive = Overestimation, Negative = Underestimation)', 
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Error distributions saved to {RESULTS_DIR}/error_distribution.png")

# ============================================================
# ANALYSIS
# ============================================================
print("\n─── Best Model Per Dataset (by RMSE) ───\n")
for dataset in DATASETS:
    subset = results_df[results_df['Dataset'] == dataset]
    best_row = subset.loc[subset['RMSE'].idxmin()]
    print(f"  {dataset}: {best_row['Model']} (RMSE={best_row['RMSE']}, R²={best_row['R²']}, S-Score={best_row['NASA S-Score']})")

best_overall = results_df.loc[results_df['RMSE'].idxmin()]
print(f"\n─── Overall Best Single Result ───")
print(f"  {best_overall['Model']} on {best_overall['Dataset']} (RMSE={best_overall['RMSE']})")

print("\n─── Average Performance Across All Datasets ───\n")
avg_perf = results_df.groupby('Model')[['RMSE', 'R²', 'NASA S-Score']].mean()
avg_perf = avg_perf.loc[['Random Forest', 'LSTM', 'Bi-LSTM']]
print(avg_perf.round(2).to_string())

print("\n─── Dataset Difficulty Ranking (by avg RMSE across models) ───\n")
difficulty = results_df.groupby('Dataset')['RMSE'].mean().sort_values()
conditions = {'FD001': '1 cond, 1 fault', 'FD002': '6 cond, 1 fault', 
              'FD003': '1 cond, 2 faults', 'FD004': '6 cond, 2 faults'}
for rank, (ds, rmse) in enumerate(difficulty.items(), 1):
    print(f"  {rank}. {ds} ({conditions[ds]}) — Avg RMSE: {rmse:.2f}")

print("\n" + "="*70)
print("  DONE! All results saved to results/ directory.")
print("="*70)
