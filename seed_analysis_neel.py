"""
seed_analysis.py  (corregido)
─────────────────────────────
Analiza como distintas inicializaciones de la RBM afectan el aprendizaje
del estado fundamental del TFIM para tres regimenes:

  Paramagnético  : J/h =  0.25  (J > 0, fase desordenada)
  Crítico        : J/h =  1.00  (J > 0, punto de transicion)
  Antiferromag.  : J/h =  2.00  (J > 0, orden de Neel)

Convenio de NetKet verificado:
  H = -h * sum_i sigma^x_i  +  J * sum_<ij> sigma^z_i sigma^z_j
  J > 0  →  antiferromagnetico  (dominan |0101...> y |1010...>)
  J < 0  →  ferromagnetico      (dominan |0000...> y |1111...>)

Inicializaciones fisicamente motivadas
───────────────────────────────────────
1. small_random    : W~N(0,0.01), b=0, c=0          [baseline]
2. large_random    : W~N(0,0.5),  b=0, c=0          [pesos grandes]
3. uniform         : W=b=c=0                         [dist. uniforme exacta]
4. neel_informed   : dos unidades ocultas con patrones [favorece orden de Neel,
                     alternantes opuestos, una para    respeta Z2 exactamente]
                     cada estado de Neel
5. para_informed   : c=+beta                         [dist. difusa, favorece
                                                       fase paramagnética]

La inicializacion 4 es fisicamente correcta para el regimen antiferro
(J>0 grande), y perjudicial para el parametrico. La 5 tiene el efecto
contrario.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh

from rbmopt import RBM

# ══════════════════════════════════════════════════════════════════════════════
# Parametros globales
# ══════════════════════════════════════════════════════════════════════════════
N             = 14
h_field       = 1.0
n_hidden      = 10
n_epochs      = 1500
batch_size    = 100
learning_rate = 0.01
log_every     = 10
k_cd          = 5
dataset_size  = 5000
beta          = 2.0     # magnitud del sesgo en inicializaciones informadas

# Tres regimenes — todos con J > 0 (convenio NetKet)
# El regimen "ferromagnetico" en la literatura corresponde a J < 0 en NetKet,
# pero aqui estudiamos J > 0 (antiferromagnetico) para ser consistentes con
# el resto del analisis donde se ha usado J > 0.
REGIMES = {
    "Paramagnético\n$g=0.25$" : 0.25,   # J = 0.25 * h
    "Crítico\n$g=1.00$"       : 1.00,   # J = 1.00 * h
    "Antiferromag.\n$g=2.00$" : 2.00,   # J = 2.00 * h  -> orden de Neel
}

# ══════════════════════════════════════════════════════════════════════════════
# Inicializaciones
# ══════════════════════════════════════════════════════════════════════════════

def make_init_small_random(n_vis, n_hid, rng):
    return (rng.normal(0, 0.01, (n_vis, n_hid)),
            np.zeros(n_vis),
            np.zeros(n_hid))

def make_init_large_random(n_vis, n_hid, rng):
    return (rng.normal(0, 0.5, (n_vis, n_hid)),
            np.zeros(n_vis),
            np.zeros(n_hid))

def make_init_uniform(n_vis, n_hid, rng):
    """W=b=c=0: distribucion exactamente uniforme sobre todas las configs."""
    return (np.zeros((n_vis, n_hid)),
            np.zeros(n_vis),
            np.zeros(n_hid))

def make_init_neel_informed(n_vis, n_hid, rng):
    """
    Inicializacion para orden antiferromagnetico (Neel) que respeta Z2.

    Las dos configuraciones dominantes son:
        Neel A: |0101...>  (en base {0,1})
        Neel B: |1010...>

    Una sola unidad oculta con patron alternante NO funciona porque
    v @ W[:,0] tiene signo opuesto para Neel A y Neel B, lo que da
    energias libres distintas y rompe la simetria Z2.

    La solucion correcta es usar DOS unidades ocultas con patrones
    opuestos, una "detectora" de cada estado de Neel:
        W[:, 0] = [+b, -b, +b, -b, ...]  <- activada por Neel B |1010...>
        W[:, 1] = [-b, +b, -b, +b, ...]  <- activada por Neel A |0101...>

    Verificacion:
        F(Neel A) = -softplus(Neel_A @ W[:,0]) - softplus(Neel_A @ W[:,1]) - ...
                  = -softplus(-N/2*beta) - softplus(+N/2*beta) - (resto igual)
        F(Neel B) = -softplus(+N/2*beta) - softplus(-N/2*beta) - (resto igual)
        => F(Neel A) = F(Neel B)  exactamente.

    Requiere n_hid >= 2 (siempre se cumple en la practica).
    """
    if n_hid < 2:
        raise ValueError("neel_informed requiere n_hid >= 2")

    W = rng.normal(0, 0.01, (n_vis, n_hid))
    b = np.zeros(n_vis)
    c = np.zeros(n_hid)

    pattern = np.array([beta if i % 2 == 0 else -beta for i in range(n_vis)])
    W[:, 0] =  pattern   # detectora de Neel B: |1010...>
    W[:, 1] = -pattern   # detectora de Neel A: |0101...>
    return W, b, c

def make_init_para_informed(n_vis, n_hid, rng):
    """
    Sesgos ocultos positivos grandes → todas las h tienden a activarse
    independientemente de v → marginal sobre v difusa (casi uniforme).
    Favorece la fase paramagnética.
    """
    W = rng.normal(0, 0.01, (n_vis, n_hid))
    b = np.zeros(n_vis)
    c = np.full(n_hid, beta)
    return W, b, c

INITIALIZATIONS = {
    "small_random"  : (make_init_small_random,  "#4C72B0", "-"),
    "large_random"  : (make_init_large_random,  "#DD8452", "--"),
    "uniform"       : (make_init_uniform,        "#55A868", "-."),
    "neel_informed" : (make_init_neel_informed,  "#C44E52", "-"),
    "para_informed" : (make_init_para_informed,  "#8172B2", "--"),
}

INIT_LABELS = {
    "small_random"  : r"Small random  $W\sim\mathcal{N}(0,0.01)$",
    "large_random"  : r"Large random  $W\sim\mathcal{N}(0,0.5)$",
    "uniform"       : r"Uniform  $W=b=c=0$",
    "neel_informed" : r"Néel-informed  $W_{i,0{=}-W_{i,1}}=\pm\beta$ (Z$_2$-sim.)",
    "para_informed" : r"Para-informed  $c=+\beta$",
}

# ══════════════════════════════════════════════════════════════════════════════
# Setup cuantico
# ══════════════════════════════════════════════════════════════════════════════
graph      = Hypercube(length=N, n_dim=1, pbc=True)
hi         = Spin(s=1/2, N=graph.n_nodes)
all_states = hi.all_states()   # (2^N, N) en {-1, +1}

def get_ground_state(g):
    """
    Devuelve (psi_prob, dataset_bin) para g = J/h dado.
    J = g * h_field  > 0  →  antiferromagnetico en convenio NetKet.
    """
    J = g * h_field
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
# Verificacion: imprimir configs dominantes para cada regimen
# ══════════════════════════════════════════════════════════════════════════════
#print("Verificacion de configuraciones dominantes por regimen:")
#print("─" * 60)
for label, g in REGIMES.items():
    J = g * h_field
    H = Ising(hilbert=hi, graph=graph, h=h_field, J=J)
    _, eigvecs = eigsh(H.to_sparse(), k=1, which='SA')
    psi_prob   = np.abs(eigvecs[:, 0]) ** 2
    psi_prob  /= psi_prob.sum()
    top3       = np.argsort(psi_prob)[::-1][:3]
    states_bin = ((all_states + 1) / 2).astype(int)
    short = label.replace("\n", " ")
    #print(f"\n{short}  (J={J:.2f}):")
    for idx in top3:
        cfg = states_bin[idx]
        #print(f"  {cfg}  p={psi_prob[idx]:.4f}")
#print("\n" + "─" * 60 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento
# ══════════════════════════════════════════════════════════════════════════════
def train_with_init(init_fn, psi_prob, dataset_bin, seed=42):
    rng = np.random.default_rng(seed)
    rbm = RBM(n_visible=N, n_hidden=n_hidden)
    W, b, c      = init_fn(N, n_hidden, rng)
    rbm.W, rbm.b, rbm.c = W, b, c

    history = rbm.train(
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
    return np.array([e for e, _ in history]), np.array([k for _, k in history])

# Bucle principal
results = {}
for regime_label, g in REGIMES.items():
    print(f"\n{'═'*55}")
    print(f"  {regime_label.replace(chr(10),' ')}  |  g = {g:.2f}")
    print(f"{'═'*55}")

    psi_prob, dataset_bin = get_ground_state(g)
    results[regime_label] = {}

    for init_name, (init_fn, color, ls) in INITIALIZATIONS.items():
        print(f"  → {init_name}")
        epochs, kls = train_with_init(init_fn, psi_prob, dataset_bin)
        results[regime_label][init_name] = (epochs, kls)
        print(f"    KL inicial: {kls[0]:.4f}  |  KL final: {kls[-1]:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# Visualizacion
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

for col, (regime_label, g) in enumerate(REGIMES.items()):
    ax = fig.add_subplot(gs[col])

    for init_name, (init_fn, color, ls) in INITIALIZATIONS.items():
        epochs, kls = results[regime_label][init_name]
        ax.plot(epochs, kls,
                label=INIT_LABELS[init_name],
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

handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", ncol=3,
           fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.16))

fig.suptitle(
    f"Dependencia de la KL en la inicialización — TFIM (convenio NetKet)\n"
    f"$N={N}$ spins, $h={h_field}$, $n_{{\\mathrm{{hidden}}}}={n_hidden}$, "
    f"CD-{k_cd}, $\\beta={beta}$",
    fontsize=12, y=1.02
)

plt.savefig("seed_analysis_neel_TFIM.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfica guardada en seed_analysis_neel_TFIM.png")

# ══════════════════════════════════════════════════════════════════════════════
# Tabla resumen
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print(f"{'Régimen':<24} {'Inicialización':<18} {'KL inicial':>12} {'KL final':>12}")
print("═"*72)
for regime_label, regime_results in results.items():
    short = regime_label.replace("\n", " ")
    for init_name, (epochs, kls) in regime_results.items():
        print(f"{short:<24} {init_name:<18} {kls[0]:>12.4f} {kls[-1]:>12.6f}")
    print("─"*72)