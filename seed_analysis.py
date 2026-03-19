"""
seed_analysis.py
────────────────
Analiza cómo distintas inicializaciones de los pesos y sesgos de la RBM
afectan el aprendizaje de la distribución del estado fundamental del TFIM,
para tres regímenes: paramagnético (J/h=0.25), crítico (J/h=1.0) y
ferromagnético (J/h=2.0).

Inicializaciones consideradas
──────────────────────────────
1. small_random   : W~N(0,0.01), b=0, c=0          [baseline, tu original]
2. large_random   : W~N(0,0.5),  b=0, c=0          [pesos grandes, más desordenado]
3. uniform        : W=0, b=0, c=0                   [distribución uniforme exacta]
4. ferro_informed : W=0, b=+β (bias visible grande) [favorece todos los spins =1]
5. para_informed  : W=0, b=0, c=+β                 [favorece h activos, dist. difusa]

La inicialización 4 es "informada" para el régimen ferro, y perjudicial para
el paramagnético. La 5 tiene el efecto opuesto. Esto permite verificar si
la inicialización ayuda o dificulta el aprendizaje según el régimen.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh
import time

from rbmopt import RBM

# Empezamos un timer para medir el tiempo total de ejecución
start_time = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros globales
# ══════════════════════════════════════════════════════════════════════════════
N             = 20
h_field       = 1.0
n_hidden      = 10
n_epochs      = 1500
batch_size    = 100
learning_rate = 0.01
log_every     = 10
k_cd          = 5
dataset_size  = 5000
beta          = 2.0     # magnitud del sesgo en inicializaciones informadas

# Tres regímenes
REGIMES = {
    "Paramagnético\n$J/h=0.25$" : -0.25,
    "Crítico\n$J/h=1.00$"       : -1.00,
    "Ferromagnético\n$J/h=2.00$": -2.00,
}

# ══════════════════════════════════════════════════════════════════════════════
# Inicializaciones
# ══════════════════════════════════════════════════════════════════════════════
def make_init_small_random(n_vis, n_hid, rng):
    W = rng.normal(0, 0.01, (n_vis, n_hid))
    b = np.zeros(n_vis)
    c = np.zeros(n_hid)
    return W, b, c

def make_init_large_random(n_vis, n_hid, rng):
    W = rng.normal(0, 0.5, (n_vis, n_hid))
    b = np.zeros(n_vis)
    c = np.zeros(n_hid)
    return W, b, c

def make_init_uniform(n_vis, n_hid, rng):
    """W=b=c=0 → distribución uniforme sobre todas las configuraciones."""
    W = np.zeros((n_vis, n_hid))
    b = np.zeros(n_vis)
    c = np.zeros(n_hid)
    return W, b, c

def make_init_ferro_informed(n_vis, n_hid, rng):
    """
    Sesgo visible positivo grande → la RBM favorece v=1 para todos los spins,
    i.e. concentra la probabilidad en la configuración ferromagnética |↑↑...↑⟩.
    """
    W = rng.normal(0, 0.01, (n_vis, n_hid))
    b = np.full(n_vis, beta)    # empuja P(v=1) → 1
    c = np.zeros(n_hid)
    return W, b, c

def make_init_para_informed(n_vis, n_hid, rng):
    """
    Sesgos ocultos positivos grandes → todas las unidades ocultas tienden a
    activarse independientemente de v, haciendo que la distribución marginal
    sobre v sea más difusa (aproxima mejor una distribución ruidosa/uniforme).
    """
    W = rng.normal(0, 0.01, (n_vis, n_hid))
    b = np.zeros(n_vis)
    c = np.full(n_hid, beta)    # activa todas las h → marginal sobre v difusa
    return W, b, c

INITIALIZATIONS = {
    "small_random"  : (make_init_small_random,  "#4C72B0", "-"),
    "large_random"  : (make_init_large_random,  "#DD8452", "--"),
    "uniform"       : (make_init_uniform,        "#55A868", "-."),
    "ferro_informed": (make_init_ferro_informed, "#C44E52", "-"),
    "para_informed" : (make_init_para_informed,  "#8172B2", "--"),
}

# ══════════════════════════════════════════════════════════════════════════════
# Setup cuántico
# ══════════════════════════════════════════════════════════════════════════════
graph      = Hypercube(length=N, n_dim=1, pbc=True)
hi         = Spin(s=1/2, N=graph.n_nodes)
all_states = hi.all_states()    # (2^N, N) en {-1, +1}

def get_ground_state(J):
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
# Entrenamiento con una inicialización dada
# ══════════════════════════════════════════════════════════════════════════════
def train_with_init(init_fn, psi_prob, dataset_bin, seed=42):
    rng      = np.random.default_rng(seed)
    rbm      = RBM(n_visible=N, n_hidden=n_hidden)

    # Sobreescribir parámetros con la inicialización elegida
    W, b, c  = init_fn(N, n_hidden, rng)
    rbm.W, rbm.b, rbm.c = W, b, c

    history  = rbm.train(
        dataset       = dataset_bin,
        n_epochs      = n_epochs,
        batch_size    = batch_size,
        learning_rate = learning_rate,
        k             = k_cd,
        log_every     = log_every,
        kl_mode       = 'exact',
        psi_prob      = psi_prob,
        hi_states     = all_states,
    )
    epochs = np.array([e for e, _ in history])
    kls    = np.array([k for _, k in history])
    return epochs, kls

# ══════════════════════════════════════════════════════════════════════════════
# Bucle principal
# ══════════════════════════════════════════════════════════════════════════════
# results[regime_label][init_name] = (epochs, kls)
results = {}

for regime_label, J_ratio in REGIMES.items():
    J = J_ratio * h_field
    print(f"\n{'═'*60}")
    print(f"  Régimen: {regime_label.replace(chr(10),' ')}  |  J = {J:.2f}, h = {h_field}")
    print(f"{'═'*60}")

    psi_prob, dataset_bin = get_ground_state(J)
    results[regime_label] = {}

    for init_name, (init_fn, color, ls) in INITIALIZATIONS.items():
        print(f"  → Inicialización: {init_name}")
        epochs, kls = train_with_init(init_fn, psi_prob, dataset_bin)
        results[regime_label][init_name] = (epochs, kls)
        print(f"    KL inicial: {kls[0]:.4f}  |  KL final: {kls[-1]:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# Visualización — 3 subplots, uno por régimen
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

regime_labels = list(REGIMES.keys())
init_styles   = {name: (color, ls)
                 for name, (_, color, ls) in INITIALIZATIONS.items()}

init_labels = {
    "small_random"  : "Small random  $W\\sim\\mathcal{N}(0,0.01)$",
    "large_random"  : "Large random  $W\\sim\\mathcal{N}(0,0.5)$",
    "uniform"       : "Uniform  $W=b=c=0$",
    "ferro_informed": "Ferro-informed  $b=+\\beta$",
    "para_informed" : "Para-informed  $c=+\\beta$",
}

for col, regime_label in enumerate(regime_labels):
    ax = fig.add_subplot(gs[col])
    regime_results = results[regime_label]

    for init_name, (epochs, kls) in regime_results.items():
        color, ls = init_styles[init_name]
        ax.plot(epochs, kls,
                label=init_labels[init_name],
                color=color, linestyle=ls,
                linewidth=1.8, alpha=0.9)

    ax.set_title(regime_label, fontsize=12, pad=8)
    ax.set_xlabel("Época", fontsize=11)
    #ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlim(0, n_epochs)

    if col == 0:
        ax.set_ylabel(
            r"$D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\mathrm{RBM}})$",
            fontsize=11)

# Leyenda común debajo de los tres paneles
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", ncol=5,
           fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.13))

fig.suptitle(
    f"Dependencia de la KL en la inicialización — TFIM\n"
    f"$N={N}$ spins, $h={h_field}$, $n_{{\\mathrm{{hidden}}}}={n_hidden}$, "
    f"CD-{k_cd}, $\\beta={beta}$",
    fontsize=12, y=1.02
)

plt.savefig("seed_analysis_TFIM.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfica guardada en seed_analysis_TFIM.png")

# ══════════════════════════════════════════════════════════════════════════════
# Tabla resumen: KL inicial y final por régimen e inicialización
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print(f"{'Régimen':<22} {'Inicialización':<18} {'KL inicial':>12} {'KL final':>12}")
print("═"*70)
for regime_label, regime_results in results.items():
    short = regime_label.replace("\n", " ")
    for init_name, (epochs, kls) in regime_results.items():
        print(f"{short:<22} {init_name:<18} {kls[0]:>12.4f} {kls[-1]:>12.6f}")
    print("-"*70)

# ══════════════════════════════════════════════════════════════════════════════
# Tiempo total de ejecución
# ══════════════════════════════════════════════════════════════════════════════
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTiempo total de ejecución: {elapsed_time:.2f} segundos")