import sys
print(sys.version)


# In[2]:


import numpy as np


# In[3]:


import time


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[5]:


import pandas as pd


# In[6]:


import matplotlib.pyplot as plt


# ## Some tests

# #### Surrogate gradient

# In[7]:


import torch
import matplotlib.pyplot as plt
import snntorch.functional as SF


# In[8]:


import snntorch.surrogate as surrogate


# In[9]:


# Define a função spike (degrau) — apenas para visualização
def step_function(x):
    return torch.where(x > 0, 1.0, 0.0)

# Surrogate gradient — fast sigmoid
alpha = 1.0
surrogate_grad = surrogate.fast_sigmoid(slope=alpha)

# Geração de dados
x = torch.linspace(-5, 5, steps=100, requires_grad=True)
spike_out = step_function(x)                  # Output da função spike
surrogate_out = surrogate_grad(x)             # Derivada surrogate (substituta)

# Plot
plt.figure(figsize=(9, 3))

# Função spike (forward)
plt.subplot(1, 2, 1)
plt.plot(x.detach(), spike_out.detach(), label="Step function", color='blue')
plt.title("Spike Function (Forward Pass)")
plt.xlabel("Membrane Potential (V)")
plt.ylabel("Output Spike")
plt.grid(True)
plt.legend()

# Gradiente surrogate (backward)
plt.subplot(1, 2, 2)
plt.plot(x.detach(), surrogate_out.detach(), label="Surrogate Gradient", color='red')
plt.title("Surrogate Gradient (Backward Pass)")
plt.xlabel("Membrane Potential (V)")
plt.ylabel("Gradient Value")
plt.grid(True)
plt.legend(loc=(0.001,0.5) )

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# ## Real Code 

# In[49]:


import pandas as pd
import numpy as np


# In[50]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# In[51]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[54]:


import snntorch as snn
#import snntorch.functional as SF
from snntorch import surrogate


# In[55]:


import snntorch.spikeplot as splt


# ### Data preparation

# In[56]:


# 1. Carregar e preparar dados
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values


# In[57]:


X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalizar e # 4) Converter para tensores
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)


# In[59]:


batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# In[60]:


# 2. Definir a rede SNN com __init__ correto
class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps=25):
        spk2_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        spk2_rec = torch.stack(spk2_rec)  # [num_steps, batch_size, output_size]
        return spk2_rec


# In[72]:


print(model)


# In[61]:


# 3. Instanciar modelo
input_size = X_train.shape[1]  # 8 features
hidden_size = 128
output_size = 2

model = SNNModel(input_size, hidden_size, output_size)


# In[62]:


# 4. Definir otimizador e função de perda
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# In[63]:


# 5. Função para treino de uma época
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, targets in train_loader:
        optimizer.zero_grad()
        spk_rec = model(data, num_steps)
        out = spk_rec.sum(dim=0)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 6. Função para avaliar no conjunto de teste
def eval_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            spk_rec = model(data, num_steps)
            out = spk_rec.sum(dim=0)
            preds = out.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    return accuracy


# In[64]:


# 7. Loop principal de treinamento e coleta de métricas
epochs = 50
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    loss, acc = train_epoch(model, train_loader, optimizer, loss_fn)
    test_acc = eval_model(model, test_loader)

    train_losses.append(loss)
    train_accuracies.append(acc)
    test_accuracies.append(test_acc)

    interval = epochs // 5  
    if epoch % interval == 0 or epoch == epochs:
        print(f"Epoch {epoch:02d}: Loss={loss:.4f}, Train Acc={acc:.4f}, Test Acc={test_acc:.4f}")


# In[67]:


# 8. Plotar métricas após treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.savefig(f"snn_epochs_{epochs}_hidden_{hidden_size}_batch_size_{batch_size}.png", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# #### 9. Visualizar spikes para algumas amostras do teste

# In[69]:


num_to_plot = 5
fig, ax = plt.subplots(num_to_plot, 1, figsize=(10, 2*num_to_plot), sharex=True)
for i in range(num_to_plot):
    spikes_to_plot = spk_rec[:, i, :].cpu()  # [num_steps, output_size]
    splt.raster(spikes_to_plot, ax=ax[i])    # sem transpor
    ax[i].invert_yaxis()                      # opcional para visual melhor
    ax[i].set_xlabel("Neuron")
    ax[i].set_ylabel("Time step")
    ax[i].set_title(f"Sample {i} - True label: {targets_sample[i].item()}")

plt.subplots_adjust(hspace=0.5)  # aumenta o espaço vertical entre os subplots
plt.savefig(f"snn_spikes_epochs_{epochs}_hidden_{hidden_size}_batch_size_{batch_size}.png", dpi=300, bbox_inches='tight')
plt.show()


# In[66]:


# 9. Visualizar spikes para algumas amostras do teste
data_iter = iter(test_loader)
data_sample, targets_sample = next(data_iter)

model.eval()
with torch.no_grad():
    spk_rec = model(data_sample, num_steps)

num_to_plot = 5
fig, ax = plt.subplots(num_to_plot, 1, figsize=(10, 2*num_to_plot), sharex=True)
for i in range(num_to_plot):
    spikes_to_plot = spk_rec[:, i, :].cpu()
    splt.raster(spikes_to_plot.T, ax=ax[i])
    ax[i].set_title(f"Sample {i} - True label: {targets_sample[i].item()}")

plt.xlabel("Time step")
plt.savefig("snn_spikes_epochs_"+str(epochs)+"_hidden_"+str(hidden_size)+"_batch_size_"+str(batch_size)+".png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

plt.show()


# In[70]:


num_to_plot = 5
fig, ax = plt.subplots(num_to_plot, 1, figsize=(10, 2*num_to_plot), sharex=True)

for i in range(num_to_plot):
    spikes_to_plot = spk_rec[:, i, :].cpu().numpy()  # [num_steps, output_size]
    num_steps, num_neurons = spikes_to_plot.shape

    for neuron in range(num_neurons):
        spike_times = np.where(spikes_to_plot[:, neuron] > 0)[0]
        ax[i].vlines(spike_times, neuron + 0.5, neuron + 1.5, color="black")

    ax[i].set_ylim(0.5, num_neurons + 0.5)
    ax[i].invert_yaxis()
    ax[i].set_ylabel("Neuron")
    ax[i].set_xlabel("Time step")
    ax[i].set_title(f"Sample {i} - True label: {targets_sample[i].item()}")

fig.subplots_adjust(hspace=0.5)
plt.show()


# In[71]:


import numpy as np
import matplotlib.pyplot as plt

# 9. Visualizar spikes para algumas amostras do teste
data_iter = iter(test_loader)
data_sample, targets_sample = next(data_iter)

model.eval()
with torch.no_grad():
    spk_rec = model(data_sample, num_steps)  # [num_steps, batch_size, output_size]

num_to_plot = 5  # quantas amostras mostrar
fig, ax = plt.subplots(num_to_plot, 1, figsize=(10, 2 * num_to_plot), sharex=True)

for i in range(num_to_plot):
    spikes_to_plot = spk_rec[:, i, :].cpu().numpy()  # [num_steps, output_size]
    num_steps_plot, num_neurons = spikes_to_plot.shape

    for neuron in range(num_neurons):
        spike_times = np.where(spikes_to_plot[:, neuron] > 0)[0]
        ax[i].vlines(spike_times, neuron + 0.5, neuron + 1.5, color="black")

    ax[i].set_ylim(0.5, num_neurons + 0.5)
    ax[i].set_ylabel("Neuron")
    ax[i].set_title(f"Sample {i} - True label: {targets_sample[i].item()}")

plt.xlabel("Time step")
fig.subplots_adjust(hspace=0.6)

# Salvando o gráfico
plt.savefig(
    f"snn_spikes_samples{num_to_plot}_epochs{epochs}_hidden{hidden_size}_batch{batch_size}.png",
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:





# In[78]:


from torchsummary import summary #pip install torchsummary
summary(model, input_size=(1, 8))  # (channels, input_features) se for linear


# # Cuidado, depois de rodar o código abaixo, no jupyter ele não imprime mais nada em tela. 

# In[82]:


from torchsummary import summary #pip install torchsummary
import io
import sys

# Redirecionar a saída do summary para uma string
buffer = io.StringIO()
sys.stdout = buffer  # redireciona a saída padrão para o buffer

# Executa o summary
summary(model, input_size=(1, 8))  # ajuste o input_size conforme o seu modelo

# Restaura a saída padrão
sys.stdout = sys.__stdout__

# Salva no arquivo
with open("summary_modelo.txt", "w") as f:
    f.write(buffer.getvalue())

# Restaura a saída padrão
sys.stdout = sys.__stdout__


# In[ ]:





# In[76]:


# Salvar a descrição do modelo em um arquivo
with open("modelo.dat", "w") as f:
    print(model, file=f)


# In[79]:


f.close() 


# In[84]:


import sys
sys.stdout = sys.__stdout__


# In[86]:


try:
    # seu código
except Exception as e:
    print("Erro:", e)


# In[87]:


print(2+2)


# In[91]:


print("Funcionando!")


# In[ ]:





# In[ ]:





# ## Appendix

# #### Packages version

# In[18]:


import sys
import pip
import torch
import numpy
import matplotlib
import sympy 
import sklearn
import pandas as pd
import matplotlib

try:
    import snntorch
except ImportError:
    snntorch = None

try:
    import torchvision
except ImportError:
    torchvision = None

print("VERSÕES DO AMBIENTE ATUAL:")
print(f"Python      : {sys.version.split()[0]}")
print(f"Pip         : {pip.__version__}")
print(f"Torch       : {torch.__version__}")
print(f"Torchvision : {getattr(torchvision, '__version__', 'não instalado')}")
print(f"SNNtorch    : {getattr(snntorch, '__version__', 'não instalado')}")
print(f"Numpy       : {numpy.__version__}")
print(f"Matplotlib  : {matplotlib.__version__}")
print(f"Sympy       : {sympy.__version__}") 
print(f"Sklearn     : {sklearn.__version__}")  
print(f"Pandas      : {pd.__version__}") 
print(f"Matplotlib  : {matplotlib.__version__}")  




# In[11]:


import sklearn
print(sklearn.__version__)


# In[5]:


import pandas as pd
print(pd.__version__)


# In[13]:


import matplotlib
print(matplotlib.__version__)


# In[8]:


import numpy
print(f"Numpy       : {numpy.__version__}")


# In[15]:


get_ipython().system('python snntorch_pima_v1.py')


# In[73]:


pip install torchsummary


# In[ ]:
