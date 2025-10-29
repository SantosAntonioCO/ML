import numpy as np
import matplotlib.pyplot as plt

# Parâmetros LIF
beta = 0.95           # decay
V_th = 1.0            # limiar
V_reset = 0.0         # reset após spike
num_steps = 50

# Entrada de corrente (aleatória)
np.random.seed(42)
I = 0.5 * np.random.randn(num_steps) + 0.7  # corrente média positiva

# Inicializa tensão de membrana
V_m = np.zeros(num_steps)
spikes = np.zeros(num_steps)

# Evolução do LIF
for t in range(1, num_steps):
    V_m[t] = beta * V_m[t-1] + I[t]
    if V_m[t] >= V_th:
        spikes[t] = 1
        V_m[t] = V_reset  # reset após disparo

# Plot
plt.figure(figsize=(10,5))
plt.plot(V_m, label="Tensão de membrana (V_m)")
plt.plot(spikes * V_th, 'r|', markersize=15, label="Spike")
plt.plot(I, '--', label="Entrada de corrente (I[t])", alpha=0.5)
plt.axhline(V_th, color='k', linestyle='--', label="Limiar (V_th)")
plt.xlabel("Passo de tempo")
plt.ylabel("Valor")
plt.title("Exemplo de neurônio LIF (Leaky Integrate-and-Fire)")
plt.legend()
plt.show()

#***************************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parâmetros LIF
beta = 0.95
V_th = 1.0
V_reset = 0.0
num_steps = 50

# Entrada de corrente aleatória
np.random.seed(42)
I = 0.5 * np.random.randn(num_steps) + 0.7

# Inicializa tensão e spikes
V_m = np.zeros(num_steps)
spikes = np.zeros(num_steps)

# Setup do gráfico
fig, ax = plt.subplots(figsize=(10,5))
line_vm, = ax.plot([], [], label="Tensão de membrana (V_m)", color='blue')
line_I, = ax.plot([], [], '--', label="Entrada I[t]", alpha=0.5)
spike_markers, = ax.plot([], [], 'r|', markersize=15, label="Spike")
ax.axhline(V_th, color='k', linestyle='--', label="Limiar V_th")
ax.set_xlim(0, num_steps)
ax.set_ylim(-0.5, 2)
ax.set_xlabel("Passo de tempo")
ax.set_ylabel("Valor")
ax.set_title("Neurônio LIF - Leaky Integrate-and-Fire")
ax.legend()

# Função de animação
def update(frame):
    global V_m, spikes
    if frame > 0:
        V_m[frame] = beta * V_m[frame-1] + I[frame]
        if V_m[frame] >= V_th:
            spikes[frame] = 1
            V_m[frame] = V_reset
    
    line_vm.set_data(np.arange(frame+1), V_m[:frame+1])
    line_I.set_data(np.arange(frame+1), I[:frame+1])
    spike_markers.set_data(np.arange(frame+1)[spikes[:frame+1]==1], V_th*np.ones(int(spikes[:frame+1].sum())))
    return line_vm, line_I, spike_markers

anim = FuncAnimation(fig, update, frames=num_steps, interval=300, blit=True)
plt.show()

#********************************************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# =====================
# Parâmetros LIF
# =====================
beta = 0.95
V_th = 1.0
V_reset = 0.0
num_steps = 50

# Entrada de corrente aleatória
np.random.seed(42)
I = 0.5 * np.random.randn(num_steps) + 0.7

# Inicializa tensão e spikes
V_m = np.zeros(num_steps)
spikes = np.zeros(num_steps)

# Setup do gráfico
fig, ax = plt.subplots(figsize=(10,5))
line_vm, = ax.plot([], [], label="Tensão de membrana (V_m)", color='blue')
line_I, = ax.plot([], [], '--', label="Entrada I[t]", alpha=0.5)
spike_markers, = ax.plot([], [], 'r|', markersize=15, label="Spike")
ax.axhline(V_th, color='k', linestyle='--', label="Limiar V_th")
ax.set_xlim(0, num_steps)
ax.set_ylim(-0.5, 2)
ax.set_xlabel("Passo de tempo")
ax.set_ylabel("Valor")
ax.set_title("Neurônio LIF - Leaky Integrate-and-Fire")
ax.legend()

# =====================
# Função de atualização
# =====================
def update(frame):
    global V_m, spikes
    if frame > 0:
        V_m[frame] = beta * V_m[frame-1] + I[frame]
        if V_m[frame] >= V_th:
            spikes[frame] = 1
            V_m[frame] = V_reset
    
    line_vm.set_data(np.arange(frame+1), V_m[:frame+1])
    line_I.set_data(np.arange(frame+1), I[:frame+1])
    spike_markers.set_data(np.arange(frame+1)[spikes[:frame+1]==1], V_th*np.ones(int(spikes[:frame+1].sum())))
    return line_vm, line_I, spike_markers

anim = FuncAnimation(fig, update, frames=num_steps, interval=300, blit=True)

# =====================
# Salvar como GIF
# =====================
gif_writer = PillowWriter(fps=2)  # 2 frames por segundo
anim.save("lif_neuron_animation.gif", writer=gif_writer)
print("✅ GIF salvo: lif_neuron_animation.gif")

# =====================
# Salvar como MP4
# =====================
mp4_writer = FFMpegWriter(fps=2)
anim.save("lif_neuron_animation.mp4", writer=mp4_writer)
print("✅ MP4 salvo: lif_neuron_animation.mp4")

plt.close()

#************************************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import shutil

# =====================
# Parâmetros LIF
# =====================
beta = 0.95
V_th = 1.0
V_reset = 0.0
num_steps = 50

# Entrada de corrente aleatória
np.random.seed(42)
I = 0.5 * np.random.randn(num_steps) + 0.7

# Inicializa tensão e spikes
V_m = np.zeros(num_steps)
spikes = np.zeros(num_steps)

# Setup do gráfico
fig, ax = plt.subplots(figsize=(10,5))
line_vm, = ax.plot([], [], label="Tensão de membrana (V_m)", color='blue')
line_I, = ax.plot([], [], '--', label="Entrada I[t]", alpha=0.5)
spike_markers, = ax.plot([], [], 'r|', markersize=15, label="Spike")
ax.axhline(V_th, color='k', linestyle='--', label="Limiar V_th")
ax.set_xlim(0, num_steps)
ax.set_ylim(-0.5, 2)
ax.set_xlabel("Passo de tempo")
ax.set_ylabel("Valor")
ax.set_title("Neurônio LIF - Leaky Integrate-and-Fire")
ax.legend()

# =====================
# Função de atualização
# =====================
def update(frame):
    global V_m, spikes
    if frame > 0:
        V_m[frame] = beta * V_m[frame-1] + I[frame]
        if V_m[frame] >= V_th:
            spikes[frame] = 1
            V_m[frame] = V_reset
    
    line_vm.set_data(np.arange(frame+1), V_m[:frame+1])
    line_I.set_data(np.arange(frame+1), I[:frame+1])
    spike_markers.set_data(np.arange(frame+1)[spikes[:frame+1]==1], V_th*np.ones(int(spikes[:frame+1].sum())))
    return line_vm, line_I, spike_markers

anim = FuncAnimation(fig, update, frames=num_steps, interval=300, blit=True)

# =====================
# Salvar animação com fallback
# =====================
if shutil.which("ffmpeg"):
    print("FFmpeg detectado! Salvando MP4...")
    mp4_writer = FFMpegWriter(fps=2)
    anim.save("lif_neuron_animation.mp4", writer=mp4_writer)
    print("✅ MP4 salvo: lif_neuron_animation.mp4")
else:
    print("FFmpeg não detectado! Salvando GIF...")
    gif_writer = PillowWriter(fps=2)
    anim.save("lif_neuron_animation.gif", writer=gif_writer)
    print("✅ GIF salvo: lif_neuron_animation.gif")

plt.close()

#************************************************************

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

plt.figure(figsize=(10,6))
plt.axis('off')

# Caixas
plt.text(0.1, 0.8, "Dataset\n(Features)", fontsize=12, ha='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
plt.text(0.5, 0.8, "Entrada de Corrente\nI[t]", fontsize=12, ha='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
plt.text(0.5, 0.5, "Neurônio LIF\nV_m[t] = β*V_m[t-1] + I[t]", fontsize=12, ha='center', bbox=dict(facecolor='orange', edgecolor='black'))
plt.text(0.5, 0.2, "Spike Train\n(0 ou 1 ao longo do tempo)", fontsize=12, ha='center', bbox=dict(facecolor='red', edgecolor='black'))

# Setas
plt.arrow(0.18, 0.8, 0.24, 0, head_width=0.03, head_length=0.03, fc='k', ec='k')
plt.arrow(0, 0.0, 0, 0.0)  # placeholder
plt.arrow(0.5, 0.7, 0, -0.15, head_width=0.03, head_length=0.03, fc='k', ec='k')
plt.arrow(0.5, 0.45, 0, -0.15, head_width=0.03, head_length=0.03, fc='k', ec='k')

plt.title("Fluxo de dados em SNN (dataset → corrente → LIF → spikes)", fontsize=14)
plt.show()

#************************************************************************************

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parâmetros da SNN
# =========================
num_features = 5      # número de features (entradas)
num_neurons = 3       # número de neurônios na camada LIF
num_steps = 20        # passos de tempo
beta = 0.9            # decay
V_th = 1.0
V_reset = 0.0

# =========================
# Dados sintéticos (toy dataset)
# =========================
np.random.seed(42)
X = np.random.rand(num_features)  # 1 amostra com 5 features

# =========================
# Inicializa tensões e spikes
# =========================
V_m = np.zeros((num_neurons, num_steps))
spikes = np.zeros((num_neurons, num_steps))

# =========================
# Mapear features para corrente de cada neurônio
# =========================
# Exemplo simples: cada neurônio recebe soma ponderada das features
weights = np.random.rand(num_neurons, num_features)
I = weights @ X  # corrente constante para simplificar

# =========================
# Evolução ao longo dos steps
# =========================
for t in range(num_steps):
    for n in range(num_neurons):
        prev_V = V_m[n, t-1] if t > 0 else 0.0
        V_m[n, t] = beta * prev_V + I[n]
        if V_m[n, t] >= V_th:
            spikes[n, t] = 1
            V_m[n, t] = V_reset

# =========================
# Plotando a matriz de spikes
# =========================
plt.figure(figsize=(10,4))
plt.imshow(spikes, cmap='Greys', aspect='auto')
plt.xlabel("Passo de tempo")
plt.ylabel("Neurônios LIF")
plt.title("Matriz de Spikes: cada linha = neurônio, cada coluna = timestep")
plt.colorbar(label="Spike (0 ou 1)")
plt.show()

# =========================
# Plotando tensões
# =========================
plt.figure(figsize=(10,4))
for n in range(num_neurons):
    plt.plot(V_m[n], label=f"Neurônio {n+1}")
plt.axhline(V_th, color='k', linestyle='--', label="Limiar")
plt.xlabel("Passo de tempo")
plt.ylabel("Tensão de membrana V_m")
plt.title("Tensão de membrana dos neurônios LIF")
plt.legend()
plt.show()

