"""
kl_vs_Jh.py
───────────
Calcula la divergencia KL al final del entrenamiento en funcion de J/h,
para distintos valores de n_hidden. Debe aparecer un pico en J/h ~ 1
(punto critico del TFIM) que se suaviza al aumentar n_hidden.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh
import time

from rbmopt import RBM

start_time = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Parametros
# ══════════════════════════════════════════════════════════════════════════════
N             = 20       # Numero de spins
h_field       = 1.0      # Campo transversal (fijo)
n_epochs      = 3000     # Epocas de entrenamiento
batch_size    = 100      # Tamano del batch
learning_rate = 0.01     # Tasa de aprendizaje
k_cd          = 5        # Pasos de Contrastive Divergence
log_every     = 100      # Solo nos interesa el valor final; log poco frecuente
dataset_size  = 5000     # Muestras para el dataset

# Valores de J/h a explorar — densos cerca de la transicion
J_ratios = np.concatenate([
    np.linspace(0.10, 0.80, 5),   # fase paramagnetica
    np.linspace(0.85, 1.15, 13),  # zona critica (mas densa)
    np.linspace(1.20, 2.00, 5),   # fase ferromagnetica
])

# Valores de n_hidden a comparar
n_hidden_values = [5, 10, 20]

# ══════════════════════════════════════════════════════════════════════════════
# Setup cuantico
# ══════════════════════════════════════════════════════════════════════════════
graph      = Hypercube(length=N, n_dim=1, pbc=True)
hi         = Spin(s=1/2, N=graph.n_nodes)
all_states = hi.all_states()   # (2^N, N) en {-1, +1}

def get_ground_state(J):
    """Devuelve (psi_prob, dataset_bin) para un J dado."""
    H = Ising(hilbert=hi, graph=graph, h=h_field, J=J)
    _, eigvecs  = eigsh(H.to_sparse(), k=1, which='SA')
    psi0        = eigvecs[:, 0]
    psi_prob    = np.abs(psi0) ** 2
    psi_prob   /= psi_prob.sum()

    idx         = np.random.choice(len(all_states), size=dataset_size,
                                   p=psi_prob)
    dataset_bin = ((all_states[idx] + 1) / 2).astype(float)
    return psi_prob, dataset_bin

# ══════════════════════════════════════════════════════════════════════════════
# Bucle principal
# ══════════════════════════════════════════════════════════════════════════════
# results[n_hidden] = array de KL finales, una por cada J/h
results = {nh: np.full(len(J_ratios), np.nan) for nh in n_hidden_values}

for i, ratio in enumerate(J_ratios):
    J = ratio * h_field
    print(f"\n{'─'*55}")
    print(f"  J/h = {ratio:.3f}  ({i+1}/{len(J_ratios)})")

    psi_prob, dataset_bin = get_ground_state(J)

    for n_hidden in n_hidden_values:
        rbm = RBM(n_visible=N, n_hidden=n_hidden)

        history = rbm.train(
            dataset       = dataset_bin,
            n_epochs      = n_epochs,
            batch_size    = batch_size,
            learning_rate = learning_rate,
            k             = k_cd,
            log_every     = log_every,
            kl_mode       = 'samples',  # 'exact' es muy costoso para N=20
            psi_prob      = psi_prob,
            hi_states     = all_states,
        )

        kl_final = history[-1][1]
        results[n_hidden][i] = kl_final
        print(f"    n_hidden={n_hidden:2d}  |  KL final = {kl_final:.5f}")

# Guardar resultados
for n_hidden in n_hidden_values:
    np.savetxt(
        f"kl_final_nh{n_hidden}.txt",
        np.column_stack([J_ratios, results[n_hidden]]),
        header="J/h   KL_final",
        fmt=["%.4f", "%.8f"]
    )
print("\nResultados guardados.")

# ══════════════════════════════════════════════════════════════════════════════
# Visualizacion
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

colors  = cm.viridis(np.linspace(0.15, 0.85, len(n_hidden_values)))
markers = ['o', 's', '^']

for (n_hidden, color, marker) in zip(n_hidden_values, colors, markers):
    kl_vals = results[n_hidden]
    ax.plot(J_ratios, kl_vals,
            color=color, marker=marker,
            markersize=5, linewidth=1.6,
            label=f"$n_{{\\mathrm{{hidden}}}}={n_hidden}$")

# Linea vertical en la transicion de fase
ax.axvline(x=1.0, color='crimson', linestyle='--',
           linewidth=1.4, alpha=0.8, label="Transición de fase $J/h=1$")

ax.set_xlabel("$J/h$", fontsize=13)
ax.set_ylabel(
    r"$D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\mathrm{RBM}})$ final",
    fontsize=13)
ax.set_title(
    f"KL al final del entrenamiento vs $J/h$ — TFIM\n"
    f"$N={N}$ spins, $h={h_field}$, {n_epochs} épocas, CD-{k_cd}",
    fontsize=12)
ax.legend(fontsize=10, framealpha=0.85)
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.set_xlim(J_ratios[0], J_ratios[-1])

plt.tight_layout()
plt.savefig("kl_final_vs_Jh.png", dpi=150, bbox_inches="tight")
plt.show()
print("Grafica guardada en kl_final_vs_Jh.png")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTiempo total de ejecución: {elapsed_time:.2f} segundos")