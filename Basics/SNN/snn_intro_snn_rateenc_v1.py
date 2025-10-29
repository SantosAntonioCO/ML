# script_snn_qsnn_replacement.py
# Requisitos:
# pip install torch snntorch scikit-learn pandas matplotlib seaborn

import os
import random
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import snntorch as snn
from snntorch import surrogate

# --------------------------
# Configurações / Hiperparâmetros
# --------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

NUM_STEPS = 25         # como tabela 2 (steps = 25)
BATCH_SIZE = 64        # tabela 2 para QSNN tem batch 64
LR = 1e-3
EPOCHS = 30

# Número de amostras exigidas (conforme sua especificação)
TRAIN_NONFRAUD = 1000
TRAIN_FRAUD = 390
TEST_NONFRAUD = 1000
TEST_FRAUD = 101

# Paths / timestamps
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"results_snn_{timestamp}"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# 1) Carregar dataset e criar splits específicos
# --------------------------
df = pd.read_csv("../../creditcard.csv")  # certifique-se que o arquivo esteja no diretório
# As colunas típicas: V1..V28, Amount, Time, Class (30 features total)
features = df.drop(columns=["Class"]).columns.tolist()
assert len(features) == 30, "Esperado 30 features no dataset (V1..V28, Amount, Time)"

# separar fraudes e não-fraudes
df_fraud = df[df["Class"] == 1].reset_index(drop=True)
df_nonfraud = df[df["Class"] == 0].reset_index(drop=True)

# Verifica se há amostras suficientes
if len(df_fraud) < (TRAIN_FRAUD + TEST_FRAUD):
    raise ValueError("Não há amostras de fraude suficientes no CSV para os números solicitados.")
if len(df_nonfraud) < (TRAIN_NONFRAUD + TEST_NONFRAUD):
    raise ValueError("Não há amostras não-fraude suficientes no CSV para os números solicitados.")

# Amostragens (embaralha antes)
df_fraud = df_fraud.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df_nonfraud = df_nonfraud.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Construir conjunto de teste fixo
test_df = pd.concat([
    df_nonfraud.iloc[:TEST_NONFRAUD],
    df_fraud.iloc[:TEST_FRAUD]
]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Remover usados para o test set
df_nonfraud_trainpool = df_nonfraud.iloc[TEST_NONFRAUD:].reset_index(drop=True)
df_fraud_trainpool = df_fraud.iloc[TEST_FRAUD:].reset_index(drop=True)

# Construir conjunto de treino (antes da validação)
train_pool_df = pd.concat([
    df_nonfraud_trainpool.iloc[:TRAIN_NONFRAUD],
    df_fraud_trainpool.iloc[:TRAIN_FRAUD]
]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Agora split treino/val 80/20 (do train_pool)
train_df = train_pool_df.sample(frac=0.8, random_state=RANDOM_SEED).reset_index(drop=True)
val_df = train_pool_df.drop(train_df.index).reset_index(drop=True)

print("Sizes -> train_total(before split):", len(train_pool_df))
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print("Train class counts:\n", train_df["Class"].value_counts())
print("Val class counts:\n", val_df["Class"].value_counts())
print("Test class counts:\n", test_df["Class"].value_counts())

# --------------------------
# 2) Pré-processamento: StandardScaler (para entrada contínua), depois MinMax(0,1) para rate coding
# --------------------------
scaler = StandardScaler()
scaler.fit(train_df[features].values)  # ajustar somente no treino (train_df)

def prepare_xy(df_subset):
    X = scaler.transform(df_subset[features].values)  # standardized
    # Depois escalamos para [0,1] para mapear p(spike) com MinMax
    mm = MinMaxScaler(feature_range=(0.0, 1.0))
    X01 = mm.fit_transform(X)  # aqui fit em cada subset é aceitável para coding; alternativa: fit no treino e transform outros
    # Obs: para manter consistência, vamos usar mm ajustado no train para val/test também:
    return X01, df_subset["Class"].values

# Para consistência entre conjuntos, criaremos MinMaxScaler ajustado no treino
from sklearn.preprocessing import MinMaxScaler
mm_global = MinMaxScaler(feature_range=(0.0, 1.0))
X_train_raw = scaler.transform(train_df[features].values)
mm_global.fit(X_train_raw)

def prepare_xy_global(df_subset):
    X_std = scaler.transform(df_subset[features].values)
    X01 = mm_global.transform(X_std)
    return X01, df_subset["Class"].values

X_train01, y_train = prepare_xy_global(train_df)
X_val01, y_val = prepare_xy_global(val_df)
X_test01, y_test = prepare_xy_global(test_df)

# --------------------------
# 3) Rate coding -> criar spike trains bernoulli com prob = feature_value (ou scaled by factor)
#    Implementamos um Dataset que gera spike trains já (shape: [num_steps, features]).
# --------------------------
class SpikeDataset(Dataset):
    def __init__(self, X01, y, num_steps=NUM_STEPS, seed=None):
        # X01: numpy array shape [N, F] with values in [0,1]
        self.X01 = X01.astype(np.float32)
        self.y = y.astype(np.int64)
        self.num_steps = num_steps
        self.N, self.F = self.X01.shape
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        # Pre-generate spike trains (deterministic per seed) to keep reproducible
        # shape: [N, num_steps, F]
        self.spikes = self.rng.rand(self.N, self.num_steps, self.F) < self.X01[:, None, :]
        self.spikes = self.spikes.astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # returns: spikes [num_steps, F], label scalar
        return torch.from_numpy(self.spikes[idx]), torch.tensor(self.y[idx], dtype=torch.long)

# Cria datasets
train_dataset = SpikeDataset(X_train01, y_train, num_steps=NUM_STEPS, seed=RANDOM_SEED)
val_dataset   = SpikeDataset(X_val01, y_val, num_steps=NUM_STEPS, seed=RANDOM_SEED+1)
test_dataset  = SpikeDataset(X_test01, y_test, num_steps=NUM_STEPS, seed=RANDOM_SEED+2)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# 4) Modelo: conforme pedido (substituindo VQC por FCs)
#    - linear_input: 30 -> 10
#    - fcL1: 10 -> 5
#    - fcL2: 10 -> 5
#    - concat (5+5->10) -> fc_mid 10 -> fc_out 10->2 (logits)
#    - LIFs on branches to give spiking dynamics
# --------------------------
class QSNN_Replacement(nn.Module):
    def __init__(self, in_features=30, hidden_input=10, branch_size=5, mid_size=10, out_classes=2, beta=0.9):
        super().__init__()
        # classical front
        self.linear_input = nn.Linear(in_features, hidden_input)
        # two FC branches
        self.fcL1 = nn.Linear(hidden_input, branch_size)
        self.fcL2 = nn.Linear(hidden_input, branch_size)
        # LIF neurons for branches (we'll use branching LIFs)
        self.lif_branch1 = snn.Leaky(beta=beta, spike_grad=surrogate.atan())
        self.lif_branch2 = snn.Leaky(beta=beta, spike_grad=surrogate.atan())
        # after concat
        self.fc_mid = nn.Linear(branch_size*2, mid_size)
        # optionally another lif or relu
        self.lif_mid = snn.Leaky(beta=beta, spike_grad=surrogate.atan())
        # final linear to logits (we use logits so BCEWithLogitsLoss can be used)
        self.fc_out = nn.Linear(mid_size, out_classes)

    def forward(self, x_spike): 
        # x_spike: [batch, num_steps, in_features]
        batch, steps, in_f = x_spike.shape
        # init mem for LIF layers
        mem_b1 = self.lif_branch1.init_leaky()
        mem_b2 = self.lif_branch2.init_leaky()
        mem_mid = self.lif_mid.init_leaky()

        out_rec = []  # collect logits per step
        for t in range(steps):
            x_t = x_spike[:, t, :]           # [batch, in_features], binary spikes
            cur_input = self.linear_input(x_t)  # [batch, hidden_input]
            # branch outputs (as currents)
            cur_b1 = self.fcL1(cur_input)   # [batch, branch_size]
            cur_b2 = self.fcL2(cur_input)   # [batch, branch_size]
            # LIF on branches -> produce spike outputs
            spk_b1, mem_b1 = self.lif_branch1(cur_b1, mem_b1)
            spk_b2, mem_b2 = self.lif_branch2(cur_b2, mem_b2)
            # concat spikes
            concat = torch.cat([spk_b1, spk_b2], dim=1)  # [batch, branch_size*2]
            cur_mid = self.fc_mid(concat)   # [batch, mid_size]
            spk_mid, mem_mid = self.lif_mid(cur_mid, mem_mid)
            # final logits (do not apply sigmoid here)
            logits_step = self.fc_out(spk_mid)  # [batch, out_classes]
            out_rec.append(logits_step)

        # average logits over time (rate-based readout)
        out_stack = torch.stack(out_rec, dim=0)   # [steps, batch, out_classes]
        out_mean = out_stack.mean(dim=0)          # [batch, out_classes] logits
        return out_mean

# --------------------------
# 5) Instanciar modelo, loss, otimizador
# --------------------------
model = QSNN_Replacement(in_features=30).to(device)
criterion = nn.BCEWithLogitsLoss()   # aceita logits [batch,2] e targets one-hot floats
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------------
# 6) Funções utilitárias de treino/avaliação
# --------------------------
def to_onehot_tensor(labels, num_classes=2):
    # labels: tensor shape [batch] dtype long
    return nn.functional.one_hot(labels, num_classes=num_classes).float()

def evaluate(loader):
    model.eval()
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    with torch.no_grad():
        for spikes, labels in loader:
            spikes = spikes.to(device)           # [batch, steps, features]
            labels = labels.to(device)
            logits = model(spikes)               # [batch, 2]
            probs = torch.sigmoid(logits)        # map logits to probabilities for reporting
            preds = (probs[:,1] >= 0.5).long()   # predict class=1 if prob class1 >= 0.5
            y_true_all.append(labels.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
            y_prob_all.append(probs[:,1].cpu().numpy())
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)
    return y_true, y_pred, y_prob

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    # FRR = FN / (TP + FN)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
    frr = fn / (tp + fn + 1e-12) if (tp + fn) > 0 else float('nan')
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float('nan')
    return {"cm": cm, "report": report, "frr": frr, "auc": auc, "tp":int(tp), "tn":int(tn), "fp":int(fp), "fn":int(fn)}

# --------------------------
# 7) Treinamento
# --------------------------
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for spikes, labels in train_loader:
        spikes = spikes.to(device)    # [batch, steps, features]
        labels = labels.to(device)
        labels_oh = to_onehot_tensor(labels, num_classes=2).to(device)  # [batch,2] float

        optimizer.zero_grad()
        logits = model(spikes)        # [batch,2] logits (mean across steps)
        loss = criterion(logits, labels_oh)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * spikes.size(0)
        n_batches += spikes.size(0)

    train_loss = running_loss / n_batches
    train_losses.append(train_loss)

    # Validação
    model.eval()
    val_running = 0.0
    n_val = 0
    with torch.no_grad():
        for spikes_val, labels_val in val_loader:
            spikes_val = spikes_val.to(device)
            labels_val = labels_val.to(device)
            labels_val_oh = to_onehot_tensor(labels_val, num_classes=2).to(device)
            logits_val = model(spikes_val)
            loss_val = criterion(logits_val, labels_val_oh)
            val_running += loss_val.item() * spikes_val.size(0)
            n_val += spikes_val.size(0)
    val_loss = val_running / n_val
    val_losses.append(val_loss)

    print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    # salvar melhor modelo por val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()

# se salvamos melhor estado, gravar no modelo final
if best_state is not None:
    model.load_state_dict(best_state)

# salvar modelo
model_path = os.path.join(OUT_DIR, f"snn_model_replacement_{timestamp}.pt")
torch.save(model.state_dict(), model_path)
print("Modelo salvo em:", model_path)

# --------------------------
# 8) Avaliação final: treino (todo train set), validação e teste
# --------------------------
y_train_true, y_train_pred, y_train_prob = evaluate(train_loader)
y_val_true, y_val_pred, y_val_prob = evaluate(val_loader)
y_test_true, y_test_pred, y_test_prob = evaluate(test_loader)

metrics_train = compute_metrics(y_train_true, y_train_pred, y_train_prob)
metrics_val = compute_metrics(y_val_true, y_val_pred, y_val_prob)
metrics_test = compute_metrics(y_test_true, y_test_pred, y_test_prob)

print("\n--- Train Metrics ---")
print(metrics_train["report"])
print("FRR:", metrics_train["frr"], "AUC:", metrics_train["auc"])
print("CM:\n", metrics_train["cm"])

print("\n--- Val Metrics ---")
print(metrics_val["report"])
print("FRR:", metrics_val["frr"], "AUC:", metrics_val["auc"])
print("CM:\n", metrics_val["cm"])

print("\n--- Test Metrics ---")
print(metrics_test["report"])
print("FRR:", metrics_test["frr"], "AUC:", metrics_test["auc"])
print("CM:\n", metrics_test["cm"])

# --------------------------
# 9) Salvar relatórios e figuras (loss curve, confusion matrices)
# --------------------------
# Loss curve
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Val Loss")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, f"loss_curve_{timestamp}.png"), dpi=200)
plt.close()

# Confusion matrices (test and val)
def plot_cm(cm, title, fname):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()

plot_cm(metrics_train["cm"], "Confusion Matrix - Train", f"cm_train_{timestamp}.png")
plot_cm(metrics_val["cm"], "Confusion Matrix - Val", f"cm_val_{timestamp}.png")
plot_cm(metrics_test["cm"], "Confusion Matrix - Test", f"cm_test_{timestamp}.png")

# salvar classification reports e métricas
with open(os.path.join(OUT_DIR, f"classification_report_{timestamp}.txt"), "w") as f:
    f.write("=== TRAIN ===\n")
    f.write(metrics_train["report"] + "\n")
    f.write(f"FRR: {metrics_train['frr']:.6f}  AUC: {metrics_train['auc']:.6f}\n")
    f.write("CM:\n" + np.array2string(metrics_train["cm"]) + "\n\n")

    f.write("=== VAL ===\n")
    f.write(metrics_val["report"] + "\n")
    f.write(f"FRR: {metrics_val['frr']:.6f}  AUC: {metrics_val['auc']:.6f}\n")
    f.write("CM:\n" + np.array2string(metrics_val["cm"]) + "\n\n")

    f.write("=== TEST ===\n")
    f.write(metrics_test["report"] + "\n")
    f.write(f"FRR: {metrics_test['frr']:.6f}  AUC: {metrics_test['auc']:.6f}\n")
    f.write("CM:\n" + np.array2string(metrics_test["cm"]) + "\n\n")

print("Resultados e figuras salvos em:", OUT_DIR)
