import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ================================
#   Configurações
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_QUBITS = 8
N_LAYERS = 2  # número de camadas RX + CNOT
LR = 1e-2
BATCH_SIZE = 32
EPOCHS = 10

# ================================
#  Dados de exemplo (substituir pelo creditcard.csv)
# ================================
# Supondo 30 features, 1000 amostras
X = np.random.rand(1000, 30)
y = np.random.randint(0, 2, 1000)

# Normalização 0 → pi/2 para AngleEmbedding
scaler = MinMaxScaler(feature_range=(0, np.pi/2))
X_scaled = scaler.fit_transform(X)

# Treino / teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# ================================
#  Dispositivo e QNode
# ================================
dev = qml.device("default.qubit", wires=N_QUBITS)

def entangling_layer(wires):
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    qml.CNOT(wires=[wires[-1], wires[0]])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(x, thetas):
    # AngleEmbedding
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    # Camada variacional RX + entangling CNOT
    thetas = thetas.reshape(N_LAYERS, N_QUBITS)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(thetas[l, q], wires=q)
        entangling_layer(range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ================================
#  Modelo híbrido
# ================================
class HybridQMLP(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        # Camada clássica para reduzir 30→5 features
        self.fc_reduce = nn.Linear(30, n_qubits)
        # Parâmetros da camada quântica
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers * n_qubits))
        # Camada clássica final
        self.mlp = nn.Sequential(
            nn.Linear(n_qubits, 125),
            nn.ReLU(),
            nn.Linear(125, 1)  # saída logit
        )
    
    def forward(self, x_batch):
        # Reduzir features 30 → 5
        x_red = self.fc_reduce(x_batch)
        
        # Executa QNode por amostra do batch
        expvals_list = [quantum_circuit(x_red[i], self.q_weights) for i in range(x_red.shape[0])]
        # Converte cada saída em Tensor float32 e empilha
        expvals = torch.stack([torch.tensor(ev, dtype=torch.float32, device=DEVICE) for ev in expvals_list])
        
        logits = self.mlp(expvals)
        return logits.squeeze(-1)

# ================================
#  Instanciação
# ================================
model = HybridQMLP(N_QUBITS, N_LAYERS).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================================
#  Treino
# ================================
train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

# ================================
#  Avaliação
# ================================
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
