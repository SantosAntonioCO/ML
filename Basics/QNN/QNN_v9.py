#!/usr/bin/env python
# coding: utf-8

# In[1]:


# qnn_synthetic_classification.py
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim


# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)


# In[4]:


# -------------------------
# Reprodutibilidade
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[82]:


# -------------------------
# Configurações
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

N_QUBITS = 8        # número de qubits / dimensão do embedding
N_LAYERS = 5        # camadas variacionais RX + entangling
LR = 1e-2
BATCH_SIZE = 30      # como você pediu
EPOCHS = 10
N_PER_CLASS = 200    # default 10 (mude para 100, 1000, ...)
FEATURE_RANGE = (0, 2 * np.pi)  # escala para AngleEmbedding


# In[83]:


def generate_dataset(n_per_class=10):
    x0 = np.random.rand(n_per_class) * 10
    x1 = np.random.rand(n_per_class) * 10

    y0 = x0 + 3 + np.random.randn(n_per_class) * 1.0
    y1 = 2 * x1 + 10 + np.random.randn(n_per_class) * 2.0

    multipliers = np.linspace(1.0, 1.0 + 0.1 * (N_QUBITS - 1), N_QUBITS)

    def make_vector(y_scalar):
        base = y_scalar.reshape(-1, 1) * multipliers.reshape(1, -1)
        noise = np.random.randn(len(y_scalar), N_QUBITS) * 0.01
        return base + noise  # shape (n_per_class, N_QUBITS)

    X0 = make_vector(y0)
    X1 = make_vector(y1)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(len(X0)), np.ones(len(X1))]).astype(int)
    return X, y


# In[84]:


# Create dataset
X, y = generate_dataset(N_PER_CLASS)
print("Raw X shape:", X.shape, "y shape:", y.shape)


# In[85]:


# Scale to [0, 2pi] for AngleEmbedding
scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
X_scaled = scaler.fit_transform(X)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)


# In[86]:


# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# In[87]:


# -------------------------
# Definição da camada quântica (adaptada pra retornar tensor compatível)
# NAO FUNCIONA
# -------------------------
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # parâmetros treináveis: inicializar entre -pi e +pi
        # usamos torch, então definimos nn.Parameter diretamente
        init = (torch.rand(n_layers * n_qubits) * 2 * np.pi) - np.pi
        self.q_weights = nn.Parameter(init)

        # definimos o QNode interno — aqui retornamos stacked expvals (tensor compatível)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, thetas):
            # x: tensor shape (n_qubits,) com ângulos
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            thetas = thetas.reshape(self.n_layers, self.n_qubits)
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RX(thetas[l, q], wires=q)
                # entangling ring
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # retornamos um tensor empilhado para manter autograd
            return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

        self.circuit = circuit

    def forward(self, x_batch):
        """
        x_batch: tensor shape (batch_size, n_qubits)
        vamos chamar o QNode para cada amostra:
        - se circuit retorna tensor torch, então torch.stack mantém grafo
        """
        # assegura dtype float32
        x_batch = x_batch.to(torch.float32)

        # chamando o QNode por amostra — mantemos o retorno tensor do QNode
        # (ideal: vetorizar via qml.batch_input, mas mantemos esta forma para ficar próximo ao que pediu)
        expvals = torch.stack([self.circuit(x_batch[i], self.q_weights) for i in range(x_batch.shape[0])], dim=0)
        # expvals shape: (batch_size, n_qubits)
        return expvals


# In[88]:


# -------------------------
# Definição da camada quântica (adaptada pra retornar tensor compatível)
# TBM NÂO FUNCIONA 2
# -------------------------
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        self.q_weights = nn.Parameter(0.01 * (2*np.pi*torch.rand(n_layers * n_qubits) - np.pi))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, thetas):
            qml.AngleEmbedding(x, wires=range(n_qubits))
            thetas = thetas.reshape(n_layers, n_qubits)
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RX(thetas[l, q], wires=q)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x_batch):
        x_batch = x_batch.to(torch.float32)
        expvals = torch.stack(
            [self.circuit(x_batch[i], self.q_weights) for i in range(x_batch.shape[0])],
            dim=0
        )
        return expvals


# In[89]:


# -------------------------
# Definição da camada quântica (adaptada pra retornar tensor compatível)
# testando
# -------------------------
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        self.q_weights = nn.Parameter(0.01 * (2*np.pi*torch.rand(n_layers * n_qubits) - np.pi))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, thetas):
            qml.AngleEmbedding(x, wires=range(n_qubits))
            thetas = thetas.reshape(n_layers, n_qubits)
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RX(thetas[l, q], wires=q)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x_batch):
        x_batch = x_batch.to(torch.float32)
        expvals = torch.stack(
    [torch.tensor(self.circuit(x_batch[i], self.q_weights), dtype=torch.float32) 
     for i in range(x_batch.shape[0])],
    dim=0
)
        return expvals


# In[90]:


# -------------------------
# Model completo: QuantumLayer + cabeça clássica
# -------------------------
class HybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.quantum = QuantumLayer(n_qubits, n_layers)
        # final classifier (recebe expvals de dimensão n_qubits)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # logit
        )

    def forward(self, x):
        # x shape: (batch, n_qubits) already angle-scaled
        expvals = self.quantum(x)              # (batch, n_qubits)
        logits = self.classifier(expvals)      # (batch, 1)
        return logits.squeeze(-1)              # (batch,)


# In[91]:


# -------------------------
# Training loop
# -------------------------
train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[92]:


# -------------------------
# Instantiate
# -------------------------
model = HybridModel(N_QUBITS, N_LAYERS).to(DEVICE)
print("Model instantiated. #params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# In[93]:


loss_history = []
start_time = time.time()
theta_history = []  # saving theta 
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        logits = model(xb)                # forward
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    

        # progress print por batch (opcional)
        print(f"Epoch {epoch}/{EPOCHS} — Batch {batch_idx}/{len(train_loader)} — Loss: {loss.item():.4f}")
    theta_history.append(model.quantum.q_weights.detach().cpu().numpy().copy())
    avg_loss = running_loss / max(1, n_batches)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f} — time elapsed: {time.time()-start_time:.1f}s\n")


# In[95]:


import winsound

# Set frequency to 2500 Hertz
frequency = 2500
# Set duration to 1000 milliseconds (1 second)
duration = 1000

# Make beep sound
winsound.Beep(frequency, duration)


# In[96]:


# -------------------------
# Evaluation
# -------------------------
model.eval()
with torch.no_grad():
    logits_test = model(X_test_t)
    probs = torch.sigmoid(logits_test).cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_true = y_test.astype(int)

    # metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    try:
        auc_pr = average_precision_score(y_true, probs)
    except ValueError:
        auc_pr = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    frr = fn / (tp + fn + 1e-9)

    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:\n", cm)
    print(f"F1: {f1:.4f}  Recall: {recall:.4f}  Precision: {precision:.4f}  Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}  AUC-PR: {auc_pr:.4f}  FRR: {frr:.4f}")


# In[97]:


# -------------------------
# Plot loss
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(loss_history) + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.title("Training loss")
plt.grid(True)
plt.show()


# In[98]:

import matplotlib.pyplot as plt

# Pega um exemplo do batch de teste
sample_x = X_test_t[0]  # shape (n_qubits,)
trained_weights = model.quantum.q_weights.detach()  # pega pesos aprendidos

# Gera o desenho do circuito
fig, ax = qml.draw_mpl(model.quantum.circuit)(sample_x, trained_weights)
plt.show()


