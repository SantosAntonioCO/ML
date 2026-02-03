#!/usr/bin/env python
# coding: utf-8

# # Unário com cosseno 

# ## Trial 21 ✅ pyramid

# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS ortogonal (Givens real)
# ============================================================
#def RBS(theta, wires):
#    """
#    Rotation Beam Splitter ortogonal.
#    Preserva o subespaço unário.         => gera non-unitary
#    """
#    qml.IsingXX(theta, wires=wires)
#    qml.IsingYY(theta, wires=wires)



#def RBS(theta, wires):  
#    a, b = wires
#    qml.CNOT(wires=[a, b])
#    qml.RY(theta, wires=b)
#    qml.CNOT(wires=[a, b])

#def RBS(theta, wires):
#    a, b = wires
#
#    # Mapeia |10> <-> |01>
#    qml.CNOT(wires=[b, a])
#    qml.RY(theta, wires=a)
#    qml.CNOT(wires=[b, a])

#def RBS(theta, wires):
#    a, b = wires
#
#    # |10> <-> |01>, nada mais
#    qml.ctrl(qml.RY, control=b)(theta, wires=a)
#    qml.ctrl(qml.RY, control=a)(-theta, wires=b)

def RBS(theta, wires):
    #qml.FermionicSwap(theta, wires=wires)
    qml.SingleExcitation(theta, wires=wires)



# ============================================================
# 2️⃣ Pyramid ORTOGONAL COMPLETA (QViT – como na figura)
# ============================================================
def pyramid_qvit_full(thetas, wires):
    """
    Pyramid completa do QViT (arquitetura ortogonal).
    - n qubits → n-1 camadas
    - θ compartilhado por camada
    """
    n = len(wires)
    assert len(thetas) == n - 1

    for layer in range(n - 1):
        theta = thetas[layer]
        for i in range(n - layer - 1):
            RBS(theta, wires=[wires[i], wires[i + 1]])


# ============================================================
# 3️⃣ Funções de diagnóstico
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    entries = []

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)

        entries.append((bitstring, amp, prob, unary))

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    for b, amp, prob, unary in entries:
        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"
        print(
            f"{b} | amp = {amp.real:+.6f}{amp.imag:+.6f}j "
            f"| prob = {prob:.6f} | {tag}"
        )

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 4️⃣ Extração da matriz efetiva 8×8 (subespaço unário)
# ============================================================
def extract_unary_matrix(thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            pyramid_qvit_full(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 5️⃣ Circuito principal (COM estado unário inicial)
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def pyramid_circuit(thetas):
    qml.PauliX(wires[0])  # |10000000>  ← FUNDAMENTAL
    pyramid_qvit_full(thetas, wires)
    return qml.state()


# ============================================================
# 6️⃣ Execução
# ============================================================
thetas = np.linspace(0.2, 1.4, n_qubits - 1)

# 🔹 Desenho do circuito
fig, ax = qml.draw_mpl(
    pyramid_circuit,
    decimals=2,
    show_all_wires=True
)(thetas)

fig.savefig("pyramid_qvit_full.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como pyramid_qvit_full.png")

# 🔹 Estado final
state = pyramid_circuit(thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# 🔹 Matriz efetiva 8×8
U = extract_unary_matrix(thetas, n_qubits)

print("\n📐 Matriz efetiva 8×8 (subespaço unário):")
print(np.round(U.real, 4))

print("\n🔎 Teste de unitaridade no subespaço unário:")
print("UᵀU ≈ I ?", np.allclose(U.T @ U, np.eye(n_qubits), atol=1e-8))


# In[ ]:





# ## Trial 31

# 👉 juntar a pirâmide QViT correta (Givens/RBS que preserva o subespaço unário)
# 👉 com o loader unário via arccos (como no artigo)
# 👉 mantendo TODAS as funções de diagnóstico, extração de matriz e testes numéricos
# 👉 garantindo que o estado permaneça unário durante TODO o processo
# 
# Abaixo está o script completo, limpo e coerente, já consolidado com o que validamos nas últimas iterações.

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS ortogonal (preserva subespaço unário)
# ============================================================
def RBS(theta, wires):
    """
    Rotation Beam Splitter que preserva o número de excitações.
    Implementado via SingleExcitation (equivalente a Givens real).
    """
    qml.SingleExcitation(theta, wires=wires)


# ============================================================
# 2️⃣ Loader unário (via arccos, como no artigo)
# ============================================================
def rbs_loader(x):
    """
    Loader unário:
    |x> = sum_i x_i |0...010...0>
    usando cadeia de RBS e ângulos via arccos.
    """
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    n = len(x)

    # Estado inicial |10...0>
    qml.PauliX(0)

    alphas = []
    prod = 1.0

    for k in range(n - 1):
        if abs(prod) < 1e-12:
            raise ValueError("Produto nulo na recursão do loader")

        val = np.clip(x[k] / prod, -1.0, 1.0)
        alpha = np.arccos(val)
        alphas.append(alpha)
        prod *= np.sin(alpha)

    for k, alpha in enumerate(alphas):
        RBS(alpha, wires=[k, k + 1])


# ============================================================
# 3️⃣ Pirâmide QViT ortogonal completa (Givens pyramid)
# ============================================================
def pyramid_qvit_full(thetas, wires):
    """
    Pirâmide QViT ortogonal completa.
    - n qubits → n-1 camadas
    - 1 parâmetro por camada (como na figura do artigo)
    """
    n = len(wires)
    assert len(thetas) == n - 1

    for layer in range(n - 1):
        theta = thetas[layer]
        for i in range(n - layer - 1):
            RBS(theta, wires=[wires[i], wires[i + 1]])


# ============================================================
# 4️⃣ Diagnóstico de estados
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)
        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"

        print(
            f"{bitstring} | "
            f"amp = {amp.real:+.6f}{amp.imag:+.6f}j | "
            f"prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 5️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(x, thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            pyramid_qvit_full(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 6️⃣ Circuito completo: Loader + barreira + Pirâmide
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qvit_loader_plus_pyramid(x, thetas):
    rbs_loader(x)

    # barreira lógica (sem label)
    qml.Barrier(wires=wires)

    pyramid_qvit_full(thetas, wires)
    return qml.state()


# ============================================================
# 7️⃣ Execução
# ============================================================
np.random.seed(0)

x = np.random.rand(n_qubits)
thetas = np.linspace(0.2, 1.4, n_qubits - 1)

# 🔹 Plot do circuito
fig, ax = qml.draw_mpl(
    qvit_loader_plus_pyramid,
    decimals=2,
    show_all_wires=True
)(x, thetas)

fig.savefig("qvit_loader_plus_pyramid.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como qvit_loader_plus_pyramid.png")

# 🔹 Estado final
state = qvit_loader_plus_pyramid(x, thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# ============================================================
# 8️⃣ Matriz efetiva e ortogonalidade
# ============================================================
U = extract_unary_matrix(x, thetas, n_qubits)

UtU = U.T @ U
I = np.eye(n_qubits)

error = UtU - I

print("\n🔎 Diagnóstico numérico de ortogonalidade:")
print("‖UᵀU − I‖_F  =", np.linalg.norm(error, ord="fro"))
print("max |UᵀU − I| =", np.max(np.abs(error)))

print("\nPassa atol=1e-8 ?", np.allclose(UtU, I, atol=1e-8))
print("Passa atol=1e-10?", np.allclose(UtU, I, atol=1e-10))


# #### RBS as 
# 
# <code>
#     """
#     qml.IsingXX(theta, wires=wires)
#     qml.IsingYY(theta, wires=wires)
# 
# </code>    
# 

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS ortogonal (preserva subespaço unário)
# ============================================================
def RBS(theta, wires):
    """
    Rotation Beam Splitter que preserva o número de excitações.
    Implementado via SingleExcitation (equivalente a Givens real).
    qml.SingleExcitation(theta, wires=wires) 
    """
    """
    Rotation Beam Splitter:
    exp(-i θ/2 (XX + YY))
    Preserva o subespaço unário
    """
    qml.IsingXX(theta, wires=wires)
    qml.IsingYY(theta, wires=wires)



# ============================================================
# 2️⃣ Loader unário (via arccos, como no artigo)
# ============================================================
def rbs_loader(x):
    """
    Loader unário:
    |x> = sum_i x_i |0...010...0>
    usando cadeia de RBS e ângulos via arccos.
    """
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    n = len(x)

    # Estado inicial |10...0>
    qml.PauliX(0)

    alphas = []
    prod = 1.0

    for k in range(n - 1):
        if abs(prod) < 1e-12:
            raise ValueError("Produto nulo na recursão do loader")

        val = np.clip(x[k] / prod, -1.0, 1.0)
        alpha = np.arccos(val)
        alphas.append(alpha)
        prod *= np.sin(alpha)

    for k, alpha in enumerate(alphas):
        RBS(alpha, wires=[k, k + 1])


# ============================================================
# 3️⃣ Pirâmide QViT ortogonal completa (Givens pyramid)
# ============================================================
def pyramid_qvit_full(thetas, wires):
    """
    Pirâmide QViT ortogonal completa.
    - n qubits → n-1 camadas
    - 1 parâmetro por camada (como na figura do artigo)
    """
    n = len(wires)
    assert len(thetas) == n - 1

    for layer in range(n - 1):
        theta = thetas[layer]
        for i in range(n - layer - 1):
            RBS(theta, wires=[wires[i], wires[i + 1]])


# ============================================================
# 4️⃣ Diagnóstico de estados
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)
        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"

        print(
            f"{bitstring} | "
            f"amp = {amp.real:+.6f}{amp.imag:+.6f}j | "
            f"prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 5️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(x, thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            pyramid_qvit_full(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 6️⃣ Circuito completo: Loader + barreira + Pirâmide
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qvit_loader_plus_pyramid(x, thetas):
    rbs_loader(x)

    # barreira lógica (sem label)
    qml.Barrier(wires=wires)

    pyramid_qvit_full(thetas, wires)
    return qml.state()


# ============================================================
# 7️⃣ Execução
# ============================================================
np.random.seed(0)

x = np.random.rand(n_qubits)
thetas = np.linspace(0.2, 1.4, n_qubits - 1)

# 🔹 Plot do circuito
fig, ax = qml.draw_mpl(
    qvit_loader_plus_pyramid,
    decimals=2,
    show_all_wires=True
)(x, thetas)

fig.savefig("qvit_loader_plus_pyramid.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como qvit_loader_plus_pyramid.png")

# 🔹 Estado final
state = qvit_loader_plus_pyramid(x, thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# ============================================================
# 8️⃣ Matriz efetiva e ortogonalidade
# ============================================================
U = extract_unary_matrix(x, thetas, n_qubits)

UtU = U.T @ U
I = np.eye(n_qubits)

error = UtU - I

print("\n🔎 Diagnóstico numérico de ortogonalidade:")
print("‖UᵀU − I‖_F  =", np.linalg.norm(error, ord="fro"))
print("max |UᵀU − I| =", np.max(np.abs(error)))

print("\nPassa atol=1e-8 ?", np.allclose(UtU, I, atol=1e-8))
print("Passa atol=1e-10?", np.allclose(UtU, I, atol=1e-10))


# #### RBS as 
# 
# <code>
# def RBS(theta, wires):
#     """
#     Rotation Beam Splitter real (Givens rotation).
#     Implementa uma rotação SO(2) no subespaço {|10>, |01>}
#     """
#     qml.Hadamard(wires=wires[0])
#     qml.Hadamard(wires=wires[1])
# 
#     qml.CNOT(wires=[wires[0], wires[1]])
#     qml.RY(0.5*theta, wires=wires[0])
#     qml.RY(-0.5*theta, wires=wires[1])
#     qml.CNOT(wires=[wires[0], wires[1]])
# 
#     qml.Hadamard(wires=wires[0])
#     qml.Hadamard(wires=wires[1])
# 
# </code>    
# 

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS ortogonal (preserva subespaço unário)
# ============================================================
def RBS(theta, wires):
    """
    Rotation Beam Splitter real (Givens rotation).
    Implementa uma rotação SO(2) no subespaço {|10>, |01>}
    """
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])

    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(0.5*theta, wires=wires[0])
    qml.RY(-0.5*theta, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])



# ============================================================
# 2️⃣ Loader unário (via arccos, como no artigo)
# ============================================================
def rbs_loader(x):
    """
    Loader unário:
    |x> = sum_i x_i |0...010...0>
    usando cadeia de RBS e ângulos via arccos.
    """
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    n = len(x)

    # Estado inicial |10...0>
    qml.PauliX(0)

    alphas = []
    prod = 1.0

    for k in range(n - 1):
        if abs(prod) < 1e-12:
            raise ValueError("Produto nulo na recursão do loader")

        val = np.clip(x[k] / prod, -1.0, 1.0)
        alpha = np.arccos(val)
        alphas.append(alpha)
        prod *= np.sin(alpha)

    for k, alpha in enumerate(alphas):
        RBS(alpha, wires=[k, k + 1])


# ============================================================
# 3️⃣ Pirâmide QViT ortogonal completa (Givens pyramid)
# ============================================================
def pyramid_qvit_full(thetas, wires):
    """
    Pirâmide QViT ortogonal completa.
    - n qubits → n-1 camadas
    - 1 parâmetro por camada (como na figura do artigo)
    """
    n = len(wires)
    assert len(thetas) == n - 1

    for layer in range(n - 1):
        theta = thetas[layer]
        for i in range(n - layer - 1):
            RBS(theta, wires=[wires[i], wires[i + 1]])


# ============================================================
# 4️⃣ Diagnóstico de estados
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)
        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"

        print(
            f"{bitstring} | "
            f"amp = {amp.real:+.6f}{amp.imag:+.6f}j | "
            f"prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 5️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(x, thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            pyramid_qvit_full(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 6️⃣ Circuito completo: Loader + barreira + Pirâmide
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qvit_loader_plus_pyramid(x, thetas):
    rbs_loader(x)

    # barreira lógica (sem label)
    qml.Barrier(wires=wires)

    pyramid_qvit_full(thetas, wires)
    return qml.state()


# ============================================================
# 7️⃣ Execução
# ============================================================
np.random.seed(0)

x = np.random.rand(n_qubits)
thetas = np.linspace(0.2, 1.4, n_qubits - 1)

# 🔹 Plot do circuito
fig, ax = qml.draw_mpl(
    qvit_loader_plus_pyramid,
    decimals=2,
    show_all_wires=True
)(x, thetas)

fig.savefig("qvit_loader_plus_pyramid.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como qvit_loader_plus_pyramid.png")

# 🔹 Estado final
state = qvit_loader_plus_pyramid(x, thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# ============================================================
# 8️⃣ Matriz efetiva e ortogonalidade
# ============================================================
U = extract_unary_matrix(x, thetas, n_qubits)

UtU = U.T @ U
I = np.eye(n_qubits)

error = UtU - I

print("\n🔎 Diagnóstico numérico de ortogonalidade:")
print("‖UᵀU − I‖_F  =", np.linalg.norm(error, ord="fro"))
print("max |UᵀU − I| =", np.max(np.abs(error)))

print("\nPassa atol=1e-8 ?", np.allclose(UtU, I, atol=1e-8))
print("Passa atol=1e-10?", np.allclose(UtU, I, atol=1e-10))


# <code>
# def RBS(theta, wires):
#     a, b = wires
# 
#     qml.Hadamard(a)
#     qml.Hadamard(b)
# 
#     qml.CNOT(wires=[a, b])
#     qml.RY(+theta / 2, wires=a)
#     qml.RY(-theta / 2, wires=b)
#     qml.CNOT(wires=[a, b])
# 
#     qml.Hadamard(a)
#     qml.Hadamard(b)
# 
# </code>    

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS ortogonal (preserva subespaço unário)
# ============================================================
def RBS(theta, wires):
    a, b = wires

    qml.Hadamard(a)
    qml.Hadamard(b)

    qml.CNOT(wires=[a, b])
    qml.RY(+theta / 2, wires=a)
    qml.RY(-theta / 2, wires=b)
    qml.CNOT(wires=[a, b])

    qml.Hadamard(a)
    qml.Hadamard(b)



# ============================================================
# 2️⃣ Loader unário (via arccos, como no artigo)
# ============================================================
def rbs_loader(x):
    """
    Loader unário:
    |x> = sum_i x_i |0...010...0>
    usando cadeia de RBS e ângulos via arccos.
    """
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    n = len(x)

    # Estado inicial |10...0>
    qml.PauliX(0)

    alphas = []
    prod = 1.0

    for k in range(n - 1):
        if abs(prod) < 1e-12:
            raise ValueError("Produto nulo na recursão do loader")

        val = np.clip(x[k] / prod, -1.0, 1.0)
        alpha = np.arccos(val)
        alphas.append(alpha)
        prod *= np.sin(alpha)

    for k, alpha in enumerate(alphas):
        RBS(alpha, wires=[k, k + 1])


# ============================================================
# 3️⃣ Pirâmide QViT ortogonal completa (Givens pyramid)
# ============================================================
def pyramid_qvit_full(thetas, wires):
    """
    Pirâmide QViT ortogonal completa.
    - n qubits → n-1 camadas
    - 1 parâmetro por camada (como na figura do artigo)
    """
    n = len(wires)
    assert len(thetas) == n - 1

    for layer in range(n - 1):
        theta = thetas[layer]
        for i in range(n - layer - 1):
            RBS(theta, wires=[wires[i], wires[i + 1]])


# ============================================================
# 4️⃣ Diagnóstico de estados
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)
        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"

        print(
            f"{bitstring} | "
            f"amp = {amp.real:+.6f}{amp.imag:+.6f}j | "
            f"prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 5️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(x, thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            pyramid_qvit_full(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 6️⃣ Circuito completo: Loader + barreira + Pirâmide
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qvit_loader_plus_pyramid(x, thetas):
    rbs_loader(x)

    # barreira lógica (sem label)
    qml.Barrier(wires=wires)

    pyramid_qvit_full(thetas, wires)
    return qml.state()


# ============================================================
# 7️⃣ Execução
# ============================================================
np.random.seed(0)

x = np.random.rand(n_qubits)
thetas = np.linspace(0.2, 1.4, n_qubits - 1)

# 🔹 Plot do circuito
fig, ax = qml.draw_mpl(
    qvit_loader_plus_pyramid,
    decimals=2,
    show_all_wires=True
)(x, thetas)

fig.savefig("qvit_loader_plus_pyramid.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como qvit_loader_plus_pyramid.png")

# 🔹 Estado final
state = qvit_loader_plus_pyramid(x, thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# ============================================================
# 8️⃣ Matriz efetiva e ortogonalidade
# ============================================================
U = extract_unary_matrix(x, thetas, n_qubits)

UtU = U.T @ U
I = np.eye(n_qubits)

error = UtU - I

print("\n🔎 Diagnóstico numérico de ortogonalidade:")
print("‖UᵀU − I‖_F  =", np.linalg.norm(error, ord="fro"))
print("max |UᵀU − I| =", np.max(np.abs(error)))

print("\nPassa atol=1e-8 ?", np.allclose(UtU, I, atol=1e-8))
print("Passa atol=1e-10?", np.allclose(UtU, I, atol=1e-10))


# <code>
# def RBS(theta, wires):
#     c = np.cos(theta)
#     s = np.sin(theta)
# 
#     U = np.array([
#         [1,  0,  0,  0],
#         [0,  c, -s,  0],
#         [0,  s,  c,  0],
#         [0,  0,  0,  1],
#     ])
# 
#     qml.QubitUnitary(U, wires=wires)
# 
# </code>    

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS ortogonal (preserva subespaço unário)
# ============================================================
def RBS(theta, wires):
    c = np.cos(theta)
    s = np.sin(theta)

    U = np.array([
        [1,  0,  0,  0],
        [0,  c, -s,  0],
        [0,  s,  c,  0],
        [0,  0,  0,  1],
    ])

    qml.QubitUnitary(U, wires=wires)


# ============================================================
# 2️⃣ Loader unário (via arccos, como no artigo)
# ============================================================
def rbs_loader(x):
    """
    Loader unário:
    |x> = sum_i x_i |0...010...0>
    usando cadeia de RBS e ângulos via arccos.
    """
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    n = len(x)

    # Estado inicial |10...0>
    qml.PauliX(0)

    alphas = []
    prod = 1.0

    for k in range(n - 1):
        if abs(prod) < 1e-12:
            raise ValueError("Produto nulo na recursão do loader")

        val = np.clip(x[k] / prod, -1.0, 1.0)
        alpha = np.arccos(val)
        alphas.append(alpha)
        prod *= np.sin(alpha)

    for k, alpha in enumerate(alphas):
        RBS(alpha, wires=[k, k + 1])


# ============================================================
# 3️⃣ Pirâmide QViT ortogonal completa (Givens pyramid)
# ============================================================
def pyramid_qvit_full(thetas, wires):
    """
    Pirâmide QViT ortogonal completa.
    - n qubits → n-1 camadas
    - 1 parâmetro por camada (como na figura do artigo)
    """
    n = len(wires)
    assert len(thetas) == n - 1

    for layer in range(n - 1):
        theta = thetas[layer]
        for i in range(n - layer - 1):
            RBS(theta, wires=[wires[i], wires[i + 1]])


# ============================================================
# 4️⃣ Diagnóstico de estados
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)
        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"

        print(
            f"{bitstring} | "
            f"amp = {amp.real:+.6f}{amp.imag:+.6f}j | "
            f"prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 5️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(x, thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            pyramid_qvit_full(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 6️⃣ Circuito completo: Loader + barreira + Pirâmide
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qvit_loader_plus_pyramid(x, thetas):
    rbs_loader(x)

    # barreira lógica (sem label)
    qml.Barrier(wires=wires)

    pyramid_qvit_full(thetas, wires)
    return qml.state()


# ============================================================
# 7️⃣ Execução
# ============================================================
np.random.seed(0)

x = np.random.rand(n_qubits)
thetas = np.linspace(0.2, 1.4, n_qubits - 1)

# 🔹 Plot do circuito
fig, ax = qml.draw_mpl(
    qvit_loader_plus_pyramid,
    decimals=2,
    show_all_wires=True
)(x, thetas)

fig.savefig("qvit_loader_plus_pyramid.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como qvit_loader_plus_pyramid.png")

# 🔹 Estado final
state = qvit_loader_plus_pyramid(x, thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# ============================================================
# 8️⃣ Matriz efetiva e ortogonalidade
# ============================================================
U = extract_unary_matrix(x, thetas, n_qubits)

UtU = U.T @ U
I = np.eye(n_qubits)

error = UtU - I

print("\n🔎 Diagnóstico numérico de ortogonalidade:")
print("‖UᵀU − I‖_F  =", np.linalg.norm(error, ord="fro"))
print("max |UᵀU − I| =", np.max(np.abs(error)))

print("\nPassa atol=1e-8 ?", np.allclose(UtU, I, atol=1e-8))
print("Passa atol=1e-10?", np.allclose(UtU, I, atol=1e-10))


# ## Butterfly

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS que preserva o subespaço unário
# ============================================================
def RBS(theta, wires):
    """
    Rotation Beam Splitter (real, subespaço unário).
    Implementado via SingleExcitation.
    """
    qml.SingleExcitation(theta, wires=wires)


# ============================================================
# 2️⃣ Reverse Butterfly ORTOGONAL (inspirado em FFT reversa)
# ============================================================
def reverse_butterfly_rbs(thetas, wires):
    """
    Reverse butterfly com RBS.
    - Preserva o subespaço unário
    - Mistura global → local
    """
    n = len(wires)
    idx = 0
    d = n // 2

    while d >= 1:
        for start in range(0, n, 2 * d):
            for i in range(d):
                RBS(
                    thetas[idx],
                    wires=[wires[start + i], wires[start + i + d]]
                )
                idx += 1
        d //= 2


# ============================================================
# 3️⃣ Funções de diagnóstico
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)

        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"
        print(
            f"{bitstring} | amp = {amp.real:+.6f}{amp.imag:+.6f}j "
            f"| prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 4️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            reverse_butterfly_rbs(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 5️⃣ Circuito principal (loader unário explícito)
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def butterfly_circuit(thetas):
    qml.PauliX(wires[0])  # |10000000>  ← estado unário fundamental
    reverse_butterfly_rbs(thetas, wires)
    return qml.state()


# ============================================================
# 6️⃣ Execução
# ============================================================
# número correto de parâmetros: (n/2) * log2(n)
n_params = (n_qubits // 2) * int(np.log2(n_qubits))
thetas = np.linspace(0.2, 1.4, n_params)

# 🔹 Desenho do circuito
fig, ax = qml.draw_mpl(
    butterfly_circuit,
    decimals=2,
    show_all_wires=True
)(thetas)

fig.savefig("reverse_butterfly_rbs.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como reverse_butterfly_rbs.png")

# 🔹 Estado final
state = butterfly_circuit(thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# 🔹 Matriz efetiva no subespaço unário
U = extract_unary_matrix(thetas, n_qubits)

print("\n📐 Matriz efetiva 8×8 (subespaço unário):")
print(np.round(U.real, 4))

print("\n🔎 Teste de ortogonalidade no subespaço unário:")
print("UᵀU ≈ I ?", np.allclose(U.T @ U, np.eye(n_qubits), atol=1e-8))

# 🔹 Erro explícito (debug fino)
err = np.linalg.norm(U.T @ U - np.eye(n_qubits))
print(f"‖UᵀU − I‖_F = {err:.3e}")


# ## Loader + Butterfly

# In[ ]:


import pennylane as qml
import numpy as np

# ============================================================
# 1️⃣ RBS que preserva o subespaço unário
# ============================================================
def RBS(theta, wires):
    """
    Rotation Beam Splitter (real, subespaço unário).
    Implementado via SingleExcitation.
    """
    qml.SingleExcitation(theta, wires=wires)


# ============================================================
# 2️⃣ Loader unário (via arccos, como no artigo)
# ============================================================
def rbs_loader(x):
    """
    Loader unário:
    |x> = sum_i x_i |0...010...0>
    usando cadeia de RBS e ângulos via arccos.
    """
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x)

    n = len(x)

    # Estado inicial |10...0>
    qml.PauliX(0)

    alphas = []
    prod = 1.0

    for k in range(n - 1):
        if abs(prod) < 1e-12:
            raise ValueError("Produto nulo na recursão do loader")

        val = np.clip(x[k] / prod, -1.0, 1.0)
        alpha = np.arccos(val)
        alphas.append(alpha)
        prod *= np.sin(alpha)

    for k, alpha in enumerate(alphas):
        RBS(alpha, wires=[k, k + 1])

    # barreira lógica (sem label)
    qml.Barrier(wires=wires)

# ============================================================
# 3️⃣ Reverse Butterfly ORTOGONAL
# ============================================================
def reverse_butterfly_rbs(thetas, wires):
    """
    Reverse butterfly com RBS.
    Mistura global → local.
    """
    n = len(wires)
    idx = 0
    d = n // 2

    while d >= 1:
        for start in range(0, n, 2 * d):
            for i in range(d):
                RBS(
                    thetas[idx],
                    wires=[wires[start + i], wires[start + i + d]]
                )
                idx += 1
        d //= 2


# ============================================================
# 4️⃣ Funções de diagnóstico
# ============================================================
def is_unary(bitstring):
    return bitstring.count("1") == 1


def print_state_diagnostics(state, n_qubits, tol=1e-12):
    print("\n📐 Diagnóstico do estado final\n")

    unary_prob = 0.0
    non_unary_prob = 0.0

    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob < tol:
            continue

        bitstring = format(i, f"0{n_qubits}b")
        unary = is_unary(bitstring)

        tag = "✅ UNARY" if unary else "❌ NÃO-UNARY"
        print(
            f"{bitstring} | amp = {amp.real:+.6f}{amp.imag:+.6f}j "
            f"| prob = {prob:.6f} | {tag}"
        )

        if unary:
            unary_prob += prob
        else:
            non_unary_prob += prob

    print("\n📊 Resumo:")
    print(f"Probabilidade total unária     = {unary_prob:.12f}")
    print(f"Probabilidade total não-unária = {non_unary_prob:.12f}")


def print_top_states(state, n_qubits, k=6):
    probs = []
    for i, amp in enumerate(state):
        prob = abs(amp) ** 2
        if prob > 0:
            probs.append((format(i, f"0{n_qubits}b"), prob))

    probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top {k} estados por probabilidade:")
    for b, p in probs[:k]:
        print(f"{b} → {p:.6f}")


# ============================================================
# 5️⃣ Extração da matriz efetiva no subespaço unário
# ============================================================
def extract_unary_matrix(thetas, n_qubits):
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    unary_indices = [1 << i for i in range(n_qubits)]
    U = np.zeros((n_qubits, n_qubits), dtype=complex)

    for col, idx in enumerate(unary_indices):

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.array(list(map(int, format(idx, f"0{n_qubits}b")))),
                wires=wires
            )
            reverse_butterfly_rbs(thetas, wires)
            return qml.state()

        psi = circuit()

        for row, jdx in enumerate(unary_indices):
            U[row, col] = psi[jdx]

    return U


# ============================================================
# 6️⃣ Circuito principal (loader unário + butterfly)
# ============================================================
n_qubits = 8
wires = list(range(n_qubits))
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def butterfly_circuit(x, thetas):
    rbs_loader(x)
    reverse_butterfly_rbs(thetas, wires)
    return qml.state()


# ============================================================
# 7️⃣ Execução
# ============================================================
# vetor clássico a ser carregado (exemplo)
x = np.linspace(0.1, 1.0, n_qubits)
x = x / np.linalg.norm(x)

# número correto de parâmetros do butterfly
n_params = (n_qubits // 2) * int(np.log2(n_qubits))
thetas = np.linspace(0.2, 1.4, n_params)

# 🔹 Desenho do circuito
fig, ax = qml.draw_mpl(
    butterfly_circuit,
    decimals=2,
    show_all_wires=True
)(x, thetas)

fig.savefig("reverse_butterfly_rbs_with_loader.png", dpi=300, bbox_inches="tight")
print("🖼️ Arquitetura salva como reverse_butterfly_rbs_with_loader.png")

# 🔹 Estado final
state = butterfly_circuit(x, thetas)

print_state_diagnostics(state, n_qubits)
print_top_states(state, n_qubits, k=6)

# 🔹 Matriz efetiva
U = extract_unary_matrix(thetas, n_qubits)

print("\n📐 Matriz efetiva 8×8 (subespaço unário):")
print(np.round(U.real, 4))

print("\n🔎 Teste de ortogonalidade no subespaço unário:")
print("UᵀU ≈ I ?", np.allclose(U.T @ U, np.eye(n_qubits), atol=1e-8))

err = np.linalg.norm(U.T @ U - np.eye(n_qubits))
print(f"‖UᵀU − I‖_F = {err:.3e}")


# # Sympy part

# In[ ]:


# Symbolic multiplication with SymPy

from sympy import symbols, expand, factor, Matrix

# 1️⃣ Define symbolic variables
x, y, z = symbols('x y z')

# 2️⃣ Create symbolic expressions
expr1 = x + 2*y
expr2 = y - z

# 3️⃣ Multiply expressions
product = expr1 * expr2

print("Product (unsimplified):", product)

# 4️⃣ Expand the multiplication
expanded = expand(product)
print("Expanded form:", expanded)

# 5️⃣ Factor back
factored = factor(expanded)
print("Factored form:", factored)

# 6️⃣ Symbolic matrix multiplication
A = Matrix([[x, 1], [y, z]])
B = Matrix([[2, y], [z, x]])
matrix_product = A * B

print("\nMatrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix product:\n", matrix_product)


# In[ ]:


from sympy import symbols, Matrix

# Define symbolic variables for two 2x2 matrices
a11, a12, a21, a22 = symbols('a11 a12 a21 a22')
b11, b12, b21, b22 = symbols('b11 b12 b21 b22')

# Define the matrices symbolically
A = Matrix([[a11, a12],
            [a21, a22]])

B = Matrix([[b11, b12],
            [b21, b22]])

# Perform symbolic matrix multiplication
C = A * B  # This follows standard tensorial (matrix) multiplication rules

# Display results
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nSymbolic Product C = A * B:")
print(C)


# In[ ]:


#from sympy import TensorProduct
from sympy import Matrix
from sympy.physics.quantum import TensorProduct
T = TensorProduct(A, B)


# In[ ]:


T


# In[ ]:


CNOT =  Matrix([
        [1, 0, 0, 0],  # |00> -> |00>
        [0, 1, 0, 0],  # |01> -> |01>
        [0, 0, 0, 1],  # |10> -> |11>
        [0, 0, 1, 0]   # |11> -> |10>
    ])


# In[ ]:


a11, a12, a21, a22 = symbols('c_x/2 -s_x/2 s_x/2 c_x/2')
b11, b12, b21, b22 = symbols('c_x/2 s_x/2 -s_x/2 c_x/2')
#https://docs.sympy.org/latest/modules/functions/elementary.html#sympy.functions.elementary.trigonometric.cos
a11, a12, a21, a22 = symbols('c_x/2 -s_x/2 s_x/2 c_x/2')
b11, b12, b21, b22 = symbols('c_x/2 s_x/2 -s_x/2 c_x/2')


# In[ ]:


A = Matrix([[a11, a12],
            [a21, a22]])

B = Matrix([[b11, b12],
            [b21, b22]])


# In[ ]:


from sympy import Matrix
from sympy.physics.quantum import TensorProduct
T = TensorProduct(A, B)


# In[ ]:


CNOT*T*CNOT


# In[ ]:


# Define the symbolic variable
x = symbols('x')


# In[ ]:


from sympy import cos, pi
from sympy import sin, pi


# In[ ]:


R =  Matrix([
        [1, 0, 0, 0],  # |00> -> |00>
        [0, cos(x/2), sin(x/2), 0],  # |01> -> |01>
        [0, -sin(x/2), cos(x/2), 0],  # |10> -> |11>
        [0, 0, 0, 1]   # |11> -> |10>
    ])


# In[ ]:


MR=R.T*R


# In[ ]:


import sympy as sp


# In[ ]:


expr = sp.sympify(MR)
print(expr)
from sympy import Matrix, pprint
pprint(expr)


# In[ ]:


#https://stackoverflow.com/questions/30541734/how-to-rewrite-sinx2-to-cos2x-form-in-sympy
#https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html
expr = expr.subs( sin(x/2)**2 + cos(x/2)**2, 1)


# In[ ]:


print(expr)


# In[ ]:


from sympy import Matrix, pprint
pprint(expr)


# In[ ]:


from sympy import symbols, diff, latex


# In[ ]:


# Convert to LaTeX
latex_str = latex(expr)
print(latex_str) # Output: \frac{\partial}{\partial x} \left(x^{2} + y z\right)


# In[ ]:





# In[ ]:





# In[ ]:




