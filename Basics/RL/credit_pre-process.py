import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#**********************************#
#    With traim sample limits      #
#**********************************#
print ("Stating prepare_creditcard_data.py ...")
# 1. Definir seed
SEED = 42
np.random.seed(SEED)

# 2. Variável para limitar número de amostras por classe no treino
LIMIT_SAMPLES = 5000  # ajuste conforme necessário

# 3. Ler dataset
data = pd.read_csv("creditcard.csv")

X = data.drop(columns=["Class"])
y = data["Class"]

# 4. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

# 5. Aplicar limite de amostras por classe no treino
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

# 6. Normalizar individualmente com StandardScaler
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)

# 7. Separar casos fraude/não fraude originais
train_nonfraud = X_train_scaled[y_train == 0]
train_fraud = X_train_scaled[y_train == 1]

test_nonfraud = X_test_scaled[y_test == 0]
test_fraud = X_test_scaled[y_test == 1]

# 8. Aplicar SMOTE separadamente em treino e teste
smote = SMOTE(random_state=SEED)

X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
X_test_smote, y_test_smote = smote.fit_resample(X_test_scaled, y_test)

train_nonfraud_smote = X_train_smote[y_train_smote == 0]
train_fraud_smote = X_train_smote[y_train_smote == 1]

test_nonfraud_smote = X_test_smote[y_test_smote == 0]
test_fraud_smote = X_test_smote[y_test_smote == 1]

# 9. Criar pasta para salvar
os.makedirs("extra_samples", exist_ok=True)

# 10. Salvar CSVs
pd.DataFrame(train_nonfraud).to_csv("extra_samples/train_nonfraud_original.csv", index=False)
pd.DataFrame(train_fraud).to_csv("extra_samples/train_fraud_original.csv", index=False)
pd.DataFrame(test_nonfraud).to_csv("extra_samples/test_nonfraud_original.csv", index=False)
pd.DataFrame(test_fraud).to_csv("extra_samples/test_fraud_original.csv", index=False)

pd.DataFrame(train_nonfraud_smote).to_csv("extra_samples/train_nonfraud_smote.csv", index=False)
pd.DataFrame(train_fraud_smote).to_csv("extra_samples/train_fraud_smote.csv", index=False)

train_full = pd.DataFrame(X_train_smote)
train_full["Class"] = y_train_smote
train_full.to_csv("extra_samples/train_full.csv", index=False)

test_full = pd.DataFrame(X_test_smote)
test_full["Class"] = y_test_smote
test_full.to_csv("extra_samples/test_full.csv", index=False)

print(" Arquivos gerados na pasta 'extra_samples'")
