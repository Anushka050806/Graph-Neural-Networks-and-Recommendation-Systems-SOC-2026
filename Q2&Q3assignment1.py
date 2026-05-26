import sys
import pickle
import numpy as np
from types import ModuleType

# --- STEP 1: SAFE COMPATIBILITY LAYER ---
# This safely handles older pickle files without breaking Seaborn's internals
if 'numpy._core' not in sys.modules:
    fake_core = ModuleType('numpy._core')
    fake_core.numeric = np.core.numeric
    sys.modules['numpy._core'] = fake_core

if 'numpy.core.numeric' not in sys.modules:
    sys.modules['numpy.core.numeric'] = np.core.numeric

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 2: LOAD DATA & PREVENT FRAGMENTATION ---
file_path = r"C:\Users\VICTUS\OneDrive\Desktop\miniconda\submission_data.pkl"

with open(file_path, "rb") as f:
    # .copy() defragments the DataFrame memory block right at the start
    df = pd.DataFrame(pickle.load(f)).copy()

print("--- Data Inspection ---")
print(f"Dataset Shape: {df.shape}")
print(f"Total Missing Values: {df.isnull().sum().sum()}")
print(df.columns)


# --- STEP 3: TARGET LABELING & LEAKAGE PREVENTION ---
# Create binary classification target (1 for Late, 0 for On-Time/Early)
df['target'] = (df['Submission_Delay'] > 0).astype(int)
#astype(int) converts the boolean Series to integers (True=1, False=0)

df = df.drop('Submission_Delay',axis=1)

# Isolate the 100 anonymous feature columns
feature_cols = [col for col in df.columns if col.startswith('x_')]



# --- STEP 5: PURE PANS MANIPULATION (NORMALIZATION) ---
# Manual Standard Scaling: (x - mean) / standard_deviation
df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

all_cols = feature_cols + ['target']
correlation_matrix = df[all_cols].corr(method='pearson')
target_corr = correlation_matrix['target'].drop('target')
top_15_features = target_corr.abs().sort_values(ascending=False).head(15).index.tolist()

print("--- Top 15 Features with Strongest Target Correlation ---")
for idx, col in enumerate(top_15_features, 1):
    raw_corr = target_corr[col]
    print(f"{idx}. {col}: {raw_corr:.4f} (Absolute: {abs(raw_corr):.4f})")

# --- STEP 3: PLOT LOCALIZED CORRELATION HEATMAP ---
# Create a matrix of just the top 15 features + target
localized_matrix = df[top_15_features + ['target']].corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(
    localized_matrix, 
    annot=True,             # Display the numerical values inside the cells
    fmt=".2f",              # Round numbers to 2 decimal places
    cmap="coolwarm",        # Blue for negative, Red for positive correlation
    vmin=-1, vmax=1,        # Lock the scale limits between -1 and 1
    linewidths=0.5,
    square=True
)
plt.title("Localized Correlation Heatmap (Top 15 Features & Target)", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


from sklearn.manifold import TSNE
import umap

# Ensure features and target variables from previous steps are ready
# feature_cols: list of all 100 'x_' columns
# top_15_features: list of the top 15 correlated columns
X_all = df[feature_cols].values
X_top15 = df[top_15_features].values
y = df['target'].values

print("--- Running Dimensionality Reduction ---")
print("This might take a few moments to compute...")

# --- 1. Compute Projections for All 100 Features ---
print("Computing t-SNE (100 features)...")
tsne_all = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne_all = tsne_all.fit_transform(X_all)

print("Computing UMAP (100 features)...")
umap_all = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap_all = umap_all.fit_transform(X_all)

# --- 2. Compute UMAP Projection for Top 15 Features ---
print("Computing UMAP (Top 15 features)...")
umap_15 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap_15 = umap_15.fit_transform(X_top15)

# --- 3. PLOT THE RESULTS SIDE-BY-SIDE ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors = ['#1f77b4', '#d62728'] # Blue for On-Time (0), Red for Late (1)
labels = ['On-Time / Early', 'Late']

# Plot A: t-SNE (All 100 Features)
for target_val, color, label in zip([0, 1], colors, labels):
    idx = (y == target_val)
    axes[0].scatter(X_tsne_all[idx, 0], X_tsne_all[idx, 1], c=color, label=label, alpha=0.5, s=15)
axes[0].set_title("t-SNE: All 100 Features", fontsize=12)
axes[0].set_xlabel("t-SNE Dimension 1")
axes[0].set_ylabel("t-SNE Dimension 2")
axes[0].legend()

# Plot B: UMAP (All 100 Features)
for target_val, color, label in zip([0, 1], colors, labels):
    idx = (y == target_val)
    axes[1].scatter(X_umap_all[idx, 0], X_umap_all[idx, 1], c=color, label=label, alpha=0.5, s=15)
axes[1].set_title("UMAP: All 100 Features", fontsize=12)
axes[1].set_xlabel("UMAP Dimension 1")
axes[1].set_ylabel("UMAP Dimension 2")
axes[1].legend()

# Plot C: UMAP (Top 15 Features)
for target_val, color, label in zip([0, 1], colors, labels):
    idx = (y == target_val)
    axes[2].scatter(X_umap_15[idx, 0], X_umap_15[idx, 1], c=color, label=label, alpha=0.5, s=15)
axes[2].set_title("UMAP: Top 15 Features Only", fontsize=12)
axes[2].set_xlabel("UMAP Dimension 1")
axes[2].set_ylabel("UMAP Dimension 2")
axes[2].legend()

plt.suptitle("Dimensionality Reduction & Geometric Signatures Exploration", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()







#Q3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Ensure variables from previous steps are accessible:
# feature_cols (all 100 features), top_15_features (top 15 features), y (target)

# Define a helper function to evaluate models and return metrics
def evaluate_models(X_data, y_data, feature_set_name):
    # Split into 80% Train and 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Initialize models
    # Note: probability=True is required for SVC to calculate AUC-ROC scores safely
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Machine (SVC)": SVC(random_state=42, probability=True),
        "Random Forest Classifier": RandomForestClassifier(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict classes and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Extract classification report metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        accuracy = report['accuracy']
        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']
        auc_roc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": f"{accuracy:.4f}",
            "Precision (Class 1)": f"{precision_1:.4f}",
            "Recall (Class 1)": f"{recall_1:.4f}",
            "F1-Score (Class 1)": f"{f1_1:.4f}",
            "AUC_ROC": f"{auc_roc:.4f}"
        })
        
    # Convert to DataFrame to print cleanly as Markdown
    results_df = pd.DataFrame(results)
    print(f"\n### Evaluation Metrics Table ({feature_set_name})")
    print(results_df.to_string(index=False))
    return results_df

# --- Task 1: Evaluate using Top 15 Features ---
X_top15 = df[top_15_features].values
metrics_top15 = evaluate_models(X_top15, y, "Top 15 Features")

# --- Task 2: Evaluate using All 100 Features (The 'More Features?' Test) ---
X_all = df[feature_cols].values
metrics_all = evaluate_models(X_all, y, "All 100 Features")




























