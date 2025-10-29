import time
from datetime import datetime

start = time.time()
print("Inicio: ", datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss'))

######################################
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import snntorch as snn
from pathlib import Path
import platform
import socket

# 1. Carregando dados e balanceando
print("1. Carregando dados e balanceando")
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_trainval, X_test, y_trainval, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# 2. Modelo
class SNNLSTMNet(nn.Module):
    def __init__(self, input_size, lstm_hidden=64, fc1=16, fc2=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        self.lif = snn.Leaky(beta=0.9)
        self.fc1 = nn.Linear(lstm_hidden, fc1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, 1)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 10, 1)
        lstm_out, _ = self.lstm(x)
        z = lstm_out[:, -1, :]
        self.lif.reset_mem()
        spk, _ = self.lif(5 * z)
        x = F.relu(self.fc1(spk))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x.squeeze()

# 3. Inicializa modelo
print("3. Inicializa modelo, loss, optimizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNLSTMNet(input_size=X.shape[1]).to(device)

n_total = len(y_train)
n_fraud = np.sum(y_train == 1)
n_normal = n_total - n_fraud
pos_weight = torch.tensor(n_normal / n_fraud, dtype=torch.float32).to(device)
print(f"Pos weight: {pos_weight.item():.2f}")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Treinamento
print("4. Treinamento")
epochs = 5
train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# 5. Avaliação (Test set)
print("5. Avaliação final (test set)")
model.eval()
y_true, y_probs = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        y_probs.extend(probs.cpu().numpy())
        y_true.extend(yb.cpu().numpy())

threshold = 0.3
y_pred_bin = [1 if p > threshold else 0 for p in y_probs]

# Save modelo
time_name = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')
log_dir = Path("results"); log_dir.mkdir(parents=True, exist_ok=True)
model_path = log_dir / f"snn_lstm_model_{time_name}.pth"
torch.save(model.state_dict(), model_path)

# Métricas e relatório
end = time.time()
avg_epoch_time = (end - start) / epochs
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
hostname = socket.gethostname()
system_info = f"{platform.system()} {platform.release()}"
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Log
log_text = f"""
===== SNN + LSTM - DETECÇÃO DE FRAUDES =====

📅 Data/Hora...............: {time_name}
💾 Modelo salvo em.........: {model_path.name}

📊 Hiperparâmetros
----------------------------
Épocas.....................: {epochs}
Batch size..................: 64
Optimizer...................: Adam
Learning rate...............: 0.001
Dropout.....................: {model.dropout.p}
LSTM hidden size............: {model.lstm.hidden_size}
FC1 out.....................: {model.fc1.out_features}
FC2 out.....................: {model.fc2.out_features}
Pesos positivos BCE.........: {pos_weight.item():.4f}
Parâmetros treináveis.......: {num_params}

📈 Avaliação Final (TESTE)
----------------------------
Threshold...................: {threshold}
F1 Score....................: {f1_score(y_true, y_pred_bin):.6f}
Precision...................: {precision_score(y_true, y_pred_bin):.6f}
Recall......................: {recall_score(y_true, y_pred_bin):.6f}
AUC.........................: {roc_auc_score(y_true, y_probs):.6f}

⏱️ Tempo
----------------------------
Tempo total de treino.......: {end - start:.2f} s
Tempo médio por época.......: {avg_epoch_time:.2f} s

🧠 Sistema
----------------------------
Dispositivo usado...........: {device_name}
Sistema Operacional.........: {system_info}
Host.........................: {hostname}
"""

log_path = log_dir / f"snn_lstm_log_{time_name}.txt"
with open(log_path, "w", encoding='utf-8') as f:
    f.write(log_text)

print(f"\n✅ Relatório salvo em: {log_path}")
print(f"✅ Modelo salvo em: {model_path}")

# Plotagens
plt.figure(); plt.plot(train_losses, marker='o'); plt.title("Loss por Época"); plt.grid(True); plt.tight_layout(); plt.savefig("train_loss_curve.png"); plt.show()
fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.4f}"); plt.plot([0, 1], [0, 1], 'k--'); plt.title("Curva ROC"); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig("roc_curve.png"); plt.show()
plt.figure(); plt.hist([p for i, p in enumerate(y_probs) if y_true[i]==0], bins=50, alpha=0.6, label="Classe 0"); plt.hist([p for i, p in enumerate(y_probs) if y_true[i]==1], bins=50, alpha=0.6, label="Classe 1"); plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold}"); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig("prob_distrib.png"); plt.show()

######################################
end1 = time.time()
print("Fim: ", datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss'))
print(f"Tempo total: {end1 - start:.2f} segundos")
