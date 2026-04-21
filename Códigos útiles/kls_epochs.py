import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube
from scipy.sparse.linalg import eigsh

from rbmopt import RBM

# ──────────────────────────────────────────────
# Parámetros globales
# ──────────────────────────────────────────────
N             = 14      # Número de spins
h             = 1.0     # Campo transversal (fijo)
n_hidden      = 20      # Neuronas ocultas
n_epochs      = 5000    # Épocas de entrenamiento
batch_size    = 100     # Tamaño del batch
learning_rate = 0.01    # Tasa de aprendizaje
log_every     = 100      # Cada cuántas épocas se registra la KL
k_cd          = 5       # Pasos de Contrastive Divergence
dataset_size  = 5000    # Muestras para estimar p_data (modo 'samples')

# Valores de J/h a explorar
J_values = [0.25, 0.9, 0.95, 1.0, 1.05, 1.1, 1.75]

# ──────────────────────────────────────────────
# Setup del sistema cuántico (común para todos los J)
# ──────────────────────────────────────────────
graph = Hypercube(length=N, n_dim=1, pbc=True)
hi    = Spin(s=1/2, N=graph.n_nodes)

# Para kl_mode='exact' necesitamos todas las configuraciones
all_states = hi.all_states()   # shape (2^N, N), en {-1, +1}

# ──────────────────────────────────────────────
# Función para obtener estado fundamental
# ──────────────────────────────────────────────
def get_ground_state(J):
    """Devuelve (psi_prob, dataset_bin) para un J dado."""
    H = Ising(hilbert=hi, graph=graph, h=h, J=J)
    eigvals, eigvecs = eigsh(H.to_sparse(), k=1, which='SA')
    psi0     = eigvecs[:, 0]
    psi_prob = np.abs(psi0) ** 2
    psi_prob = psi_prob / psi_prob.sum()

    # Dataset muestreado
    indices     = np.random.choice(len(all_states), size=dataset_size,
                                   p=psi_prob)
    dataset_pm  = all_states[indices]
    dataset_bin = ((dataset_pm + 1) / 2).astype(float)  # {-1,+1} → {0,1}

    return psi_prob, dataset_bin

# ──────────────────────────────────────────────
# Bucle principal
# ──────────────────────────────────────────────
results = {}

for J in J_values:
    ratio = J / h
    print(f"\n{'='*55}")
    print(f"  Entrenando RBM  |  J/h = {ratio:.2f}")
    print(f"{'='*55}")

    psi_prob, dataset_bin = get_ground_state(J)

    # RBM fresca para cada valor de J
    rbm = RBM(n_visible=N, n_hidden=n_hidden)

    kl_history = rbm.train(
        dataset       = dataset_bin,
        n_epochs      = n_epochs,
        batch_size    = batch_size,
        learning_rate = learning_rate,
        k             = k_cd,
        log_every     = log_every,
        kl_mode       = 'exact',     # cambiar a 'samples' para mayor velocidad
        psi_prob      = psi_prob,
        hi_states     = all_states,
    )

    results[ratio] = kl_history

    for epoch, kl in kl_history:
        print(f"  Época {epoch:5d} | KL = {kl:.6f}")

    epochs_arr = np.array([e for e, _ in kl_history])
    kl_arr     = np.array([k for _, k in kl_history])
    np.savetxt(f"kl_Jh{ratio:.2f}.txt",
               np.column_stack([epochs_arr, kl_arr]),
               header="epoch  KL", fmt=["%.0f", "%.8f"])

# ──────────────────────────────────────────────
# Visualización
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

ratios = sorted(results.keys())
colors = cm.plasma(np.linspace(0.1, 0.9, len(ratios)))

for ratio, color in zip(ratios, colors):
    history = results[ratio]
    epochs  = [e for e, _ in history]
    kls     = [k for _, k in history]
    ax.plot(epochs, kls,
            label=f"$J/h = {ratio:.2f}$",
            color=color, linewidth=1.8, alpha=0.9)

ax.set_xlabel("Época", fontsize=13)
ax.set_ylabel(r"$D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\mathrm{RBM}})$",
              fontsize=13)
ax.set_title(
    f"Evolución de la divergencia KL — TFIM\n"
    f"$N={N}$ spins, $h={h}$, "
    f"$n_{{\\mathrm{{hidden}}}}={n_hidden}$, CD-{k_cd}",
    fontsize=13
)
ax.legend(fontsize=10, loc="upper right", framealpha=0.85,
          title="$J/h$", title_fontsize=10)
#ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.set_xlim(0, n_epochs)

plt.tight_layout()
plt.savefig("kl_vs_epochs_TFIM.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfica guardada en kl_vs_epochs_TFIM.png")
