import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("Starting prepare_creditcard_data.py ...")

# ================================================
#  CONFIGS
# ================================================
SEED = 42
np.random.seed(SEED)


LIMIT_SAMPLES = 300000     # limitar tamanho do treino
print("LIMIT_SAMPLES=",LIMIT_SAMPLES)
APPLY_SMOTE_TEST = False # se True, aplica SMOTE também no teste

# ================================================
# 1. Reading full dataset 
# ================================================
# 
credit_path = "../../creditcard.csv"
data = pd.read_csv(credit_path)

X = data.drop(columns=["Class"])
y = data["Class"]

# ================================================
# 2. Split: 70% train | 15% validation | 15% test
# ================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# ================================================
# 3. Set train sample limit 
# ================================================
def limit_class_samples(X, y, limit):
    df = pd.DataFrame(X)
    df["Class"] = y.values
    limited = []
    for cls in df["Class"].unique():
        subset = df[df["Class"] == cls]
        if len(subset) > limit:
            subset = subset.sample(limit, random_state=SEED)
        limited.append(subset)
    return pd.concat(limited)

train_limited = limit_class_samples(X_train, y_train, LIMIT_SAMPLES)
X_train = train_limited.drop(columns=["Class"]).values
y_train = train_limited["Class"].values

# ================================================
# 4. Normalize each part independently.
# ================================================
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

scaler_val = StandardScaler()
X_val_scaled = scaler_val.fit_transform(X_val)

scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)

# ================================================
# 5. Divide into fraud and non-fraud.
# ================================================
def split_fraud(X, y):
    return X[y == 0], X[y == 1]

train_nonfraud, train_fraud = split_fraud(X_train_scaled, y_train)
val_nonfraud,   val_fraud   = split_fraud(X_val_scaled, y_val)
test_nonfraud,  test_fraud  = split_fraud(X_test_scaled, y_test)

# ================================================
# 6. Apply SMOTE (train ALWAYS, test OPTIONAL)
# ================================================
smote = SMOTE(random_state=SEED)

# ---- Train ----
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
train_nonfraud_smote, train_fraud_smote = split_fraud(X_train_smote, y_train_smote)

# ---- Validation ---- (SMOTE is not used in validation.)
X_val_smote = X_val_scaled.copy()
y_val_smote = y_val.values.copy()

# ---- TEST ----
if APPLY_SMOTE_TEST:
    X_test_smote, y_test_smote = smote.fit_resample(X_test_scaled, y_test)
else:
    X_test_smote = X_test_scaled.copy()
    y_test_smote = y_test.values.copy()

test_nonfraud_smote, test_fraud_smote = split_fraud(X_test_smote, y_test_smote)

# ================================================
# 7. Create dir
# ================================================
os.makedirs("extra_samples", exist_ok=True)

# ================================================
#  9. Save CSVs
# ================================================ 

# Function to save in a scrambled format
def save_shuffled(df, path):
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_shuffled.to_csv(path, index=False)

# ================================================
# 8. Save CSVs (balanced and jumbled)
# ================================================

# ---------- ORIGINAL ----------
save_shuffled(
    pd.DataFrame(train_nonfraud),
    "extra_samples/train_nonfraud_original.csv"
)
save_shuffled(
    pd.DataFrame(train_fraud),
    "extra_samples/train_fraud_original.csv"
)

save_shuffled(
    pd.DataFrame(val_nonfraud),
    "extra_samples/val_nonfraud_original.csv"
)
save_shuffled(
    pd.DataFrame(val_fraud),
    "extra_samples/val_fraud_original.csv"
)

save_shuffled(
    pd.DataFrame(test_nonfraud),
    "extra_samples/test_nonfraud_original.csv"
)
save_shuffled(
    pd.DataFrame(test_fraud),
    "extra_samples/test_fraud_original.csv"
)

# ---------- SMOTE ----------
save_shuffled(
    pd.DataFrame(train_nonfraud_smote),
    "extra_samples/train_nonfraud_smote.csv"
)
save_shuffled(
    pd.DataFrame(train_fraud_smote),
    "extra_samples/train_fraud_smote.csv"
)

# ---------- VALIDATION (SEM SMOTE) ----------
val_full = pd.DataFrame(X_val_scaled)
val_full["Class"] = y_val.values
save_shuffled(val_full, "extra_samples/validation_full.csv")

# ---------- TEST ----------
test_full = pd.DataFrame(X_test_smote)
test_full["Class"] = y_test_smote
save_shuffled(test_full, "extra_samples/test_full.csv")

# ---------- Train COMPLETO ----------
train_full = pd.DataFrame(X_train_smote)
train_full["Class"] = y_train_smote
save_shuffled(train_full, "extra_samples/train_full.csv")

print(" Balanced and scrambled files saved in 'extra_samples/'")
print("Was SMOTE applied in the TEST?" , APPLY_SMOTE_TEST)
