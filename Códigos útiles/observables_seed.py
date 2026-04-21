"""
TFIM Observable Study v3 — Inicialización física informada por datos
=====================================================================
Cambios respecto a v2
---------------------
  1. Inicialización PCA + bias desde los datos  (ver RBM_PCD.init_from_data)
       - b_i  = log(μ_i / (1−μ_i))  →  marginales visibles correctas desde epoch 0
       - W    = top-n_h vectores singulares del dataset centrado × escala
                Para g grande el 1er componente es exactamente la dirección
                ferromagnética; el modelo nace ya apuntando a la física correcta.
  2. Se elimina la gráfica de longitud de correlación.
  3. Se usa un schedule de learning rate (constante → decay) para refinar mejor.

Requiere rbm.py en el mismo directorio.
"""

import numpy as np
import netket as nk
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# 0.  Configuración global
# =============================================================================
N          = 10       # número de espines (2^10 = 1024, exacto asequible)
J          = -1.0
N_HIDDEN   = 2 * N
N_TRAIN    = 5_000
N_OBS      = 20_000   # cadenas Gibbs para observables (más → menos varianza)
N_EPOCHS   = 4_000
BATCH_SIZE = 100
LR_INIT    = 0.02     # learning rate inicial
LR_FINAL   = 0.005    # learning rate final (decay lineal)
K_CD       = 1        # PCD-1
LOG_EVERY  = 500
N_WARMUP   = 2_000    # warm-up largo para asegurar mezcla
SEED       = 42

G_VALUES = np.array([
    0.20, 0.40, 0.60,
    0.75, 0.85, 0.95,
    1.00,
    1.05, 1.15, 1.25,
    1.50, 2.00, 3.00,
])

# =============================================================================
# 1.  RBM con PCD e inicialización desde datos
# =============================================================================
from rbmopt import RBM   # RBM base (sin modificar)


class RBM_PCD(RBM):
    """RBM con PCD y arranque físicamente informado."""

    # ── Inicialización ────────────────────────────────────────────────────
    def init_from_data(self, dataset):
        """
        Inicializa los parámetros usando estadísticas del dataset.

        Sesgos visibles b
        -----------------
        Cada b_i se fija para que la marginal p(v_i = 1) del modelo iguale
        la frecuencia empírica μ_i:

            σ(b_i) = μ_i  →  b_i = log(μ_i / (1 − μ_i))

        Esto garantiza que las probabilidades marginales sean correctas desde
        el epoch 0, sin necesidad de que el entrenamiento las descubra.

        Pesos W (SVD del dataset centrado)
        -----------------------------------
        Sea X = dataset − μ  (centrado, shape M × n_visible).
        La SVD X = U S Vᵀ da los modos de varianza decreciente.
        Para g >> 1 (fase ordenada) el 1er modo singular es la dirección
        ferromagnética uniforme: (1,1,...,1)/√N, que separa las dos nubes
        de configuraciones (↑↑...↑ y ↓↓...↓).
        Inicializar W con las primeras n_hidden filas de Vᵀ hace que el
        modelo nazca ya orientado hacia esa estructura de correlación.

        Escala: se usa 0.1 × s_k/√M para que los activaciones pre-sigmoidales
        sean O(1) pero no saturen la función de activación.
        """
        M  = len(dataset)
        mu = dataset.mean(axis=0).clip(1e-3, 1 - 1e-3)   # (n_visible,)

        # Sesgos visibles
        self.b = np.log(mu / (1.0 - mu))

        # SVD del dataset centrado
        X         = dataset - mu                          # (M, n_visible)
        _, S, Vt  = np.linalg.svd(X, full_matrices=False)
        n_comp    = min(self.n_hidden, len(S))

        # Columnas de W ← vectores singulares derechos escalados
        W_svd = (Vt[:n_comp].T) * (S[:n_comp] / np.sqrt(M)) * 0.1
        if self.n_hidden > n_comp:
            extra = np.random.normal(0, 0.01,
                                     (self.n_visible, self.n_hidden - n_comp))
            self.W = np.hstack([W_svd, extra])
        else:
            self.W = W_svd

        # Sesgos ocultos — dejar a 0 (sin información a priori sobre h)
        self.c = np.zeros(self.n_hidden)

    # ── PCD ───────────────────────────────────────────────────────────────
    def train_pcd(self,
                  dataset,
                  n_epochs      = 1000,
                  batch_size    = 100,
                  lr_init       = 0.05,
                  lr_final      = None,
                  k             = 1,
                  log_every     = 10,
                  kl_mode       = 'samples',
                  psi_prob      = None,
                  hi_states     = None):
        """
        PCD-k con learning rate que decae linealmente de lr_init a lr_final.

        Diferencia clave PCD vs CD:
        Las cadenas negativas son persistentes entre mini-batches.
        Se inicializan mitad en all-1 y mitad en all-0 para cubrir
        ambos modos ferromagnéticos desde el inicio.
        """
        if kl_mode == 'exact' and (psi_prob is None or hi_states is None):
            raise ValueError("kl_mode='exact' requiere psi_prob y hi_states.")

        if lr_final is None:
            lr_final = lr_init

        n_samples       = len(dataset)
        effective_batch = min(batch_size, n_samples)
        rng             = np.random.default_rng(SEED)

        # Cadenas persistentes — arrancar mitad ↑, mitad ↓
        half        = effective_batch // 2
        chains      = np.vstack([
            np.ones( (half,                 self.n_visible), dtype=float),
            np.zeros((effective_batch-half, self.n_visible), dtype=float),
        ])

        kl_history = []

        for epoch in range(n_epochs):

            # Learning rate con decay lineal
            lr = lr_init + (lr_final - lr_init) * epoch / max(n_epochs - 1, 1)

            # Mini-batch
            idx = rng.choice(n_samples, size=effective_batch, replace=False)
            V0  = dataset[idx]

            # Fase positiva
            H0_prob = self.prob_h_given_v(V0)

            # Fase negativa — k pasos sobre cadenas persistentes
            Vk = chains
            for _ in range(k):
                Hk = self.sample_h(Vk)
                Vk = self.sample_v(Hk)
            chains = Vk.copy()

            Hk_prob = self.prob_h_given_v(Vk)

            # Gradientes
            dW = (V0.T @ H0_prob - Vk.T @ Hk_prob) / effective_batch
            db = (V0 - Vk).mean(axis=0)
            dc = (H0_prob - Hk_prob).mean(axis=0)

            self.W += lr * dW
            self.b += lr * db
            self.c += lr * dc

            if epoch % log_every == 0:
                if kl_mode == 'exact':
                    kl = self.kl_divergence_exact(psi_prob, hi_states)
                else:
                    kl = self.kl_divergence_from_samples(dataset)
                kl_history.append((epoch, kl))

        return kl_history


# =============================================================================
# 2.  Estado fundamental exacto
# =============================================================================
def exact_ground_state(N, h, J=-1.0):
    graph   = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
    hi      = nk.hilbert.Spin(s=0.5, N=N)
    H_op    = nk.operator.Ising(hilbert=hi, graph=graph, h=h, J=J)
    _, vecs = eigsh(H_op.to_sparse(), k=1, which='SA')
    prob    = np.abs(vecs[:, 0])**2
    return prob / prob.sum(), hi.all_states()


# =============================================================================
# 3.  Dataset y muestreo Gibbs
# =============================================================================
def make_dataset_01(all_states, prob, n_samples, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prob), size=n_samples, p=prob)
    return (all_states[idx] + 1.0) / 2.0       # {-1,+1} → {0,1}


def gibbs_samples_pm1(rbm, n_samples, n_warmup=500, seed=None):
    """
    n_samples cadenas paralelas (mitad ↑, mitad ↓) con n_warmup pasos.
    Devuelve (n_samples, N) en {-1, +1}.
    """
    half  = n_samples // 2
    v     = np.vstack([
        np.ones( (half,           rbm.n_visible), dtype=float),
        np.zeros((n_samples-half, rbm.n_visible), dtype=float),
    ])
    for _ in range(n_warmup):
        h = rbm.sample_h(v)
        v = rbm.sample_v(h)
    return 2.0 * v - 1.0       # {0,1} → {-1,+1}


# =============================================================================
# 4.  Observables
# =============================================================================
def observables_from_samples(states_pm1):
    """Monte-Carlo: |m| y χ desde cadenas Gibbs."""
    N_   = states_pm1.shape[1]
    m    = states_pm1.mean(axis=1)
    mag  = np.abs(m).mean()
    chi  = N_ * (np.mean(m**2) - mag**2)
    return mag, chi


def observables_exact(all_states, prob):
    """Exacto: |m| y χ desde la distribución cuántica."""
    N_   = all_states.shape[1]
    m    = all_states.mean(axis=1)
    mag  = np.dot(np.abs(m), prob)
    chi  = N_ * (np.dot(m**2, prob) - mag**2)
    return mag, chi


# =============================================================================
# 5.  Bucle principal
# =============================================================================
store = {k: [] for k in ('g',
                          'e_mag', 'e_chi',
                          'r_mag', 'r_chi',
                          'kl')}

print(f"\n{'━'*62}")
print(f"  TFIM Observable Study v3 — init. PCA + bias")
print(f"  N={N}   n_h={N_HIDDEN}   {N_EPOCHS} épocas   PCD-{K_CD}")
print(f"  lr {LR_INIT} → {LR_FINAL} (decay lineal)")
print(f"{'━'*62}")

for g in G_VALUES:
    h = abs(J) / g
    print(f"\n  g = {g:.3f}   (h = {h:.4f})")
    print(f"  {'─'*42}")

    # ── Exacto ─────────────────────────────────────────────────────
    prob, all_states = exact_ground_state(N, h, J)
    e_mag, e_chi     = observables_exact(all_states, prob)
    print(f"  Exact  → |m|={e_mag:.4f}  χ={e_chi:.4f}")

    # ── Dataset ────────────────────────────────────────────────────
    dataset = make_dataset_01(all_states, prob, N_TRAIN, seed=SEED)

    # ── RBM: crear + inicializar desde datos ───────────────────────
    rbm = RBM_PCD(n_visible=N, n_hidden=N_HIDDEN, seed=SEED)
    rbm.init_from_data(dataset)       # ← inicialización física

    # ── Entrenamiento PCD ──────────────────────────────────────────
    kl_hist = rbm.train_pcd(
        dataset,
        n_epochs  = N_EPOCHS,
        batch_size= BATCH_SIZE,
        lr_init   = LR_INIT,
        lr_final  = LR_FINAL,
        k         = K_CD,
        log_every = LOG_EVERY,
        kl_mode   = 'exact',
        psi_prob  = prob,
        hi_states = all_states,
    )
    kl_0, kl_fin = kl_hist[0][1], kl_hist[-1][1]
    arrow = '↓' if kl_fin < kl_0 else '↑'
    print(f"  KL     → {kl_0:.5f} → {kl_fin:.5f}  ({arrow})")

    # ── Muestreo y observables RBM ─────────────────────────────────
    rbm_pm1         = gibbs_samples_pm1(rbm, N_OBS, N_WARMUP, seed=SEED)
    r_mag, r_chi    = observables_from_samples(rbm_pm1)
    print(f"  RBM    → |m|={r_mag:.4f}  χ={r_chi:.4f}")

    # Acumular
    store['g'].append(g)
    store['e_mag'].append(e_mag); store['e_chi'].append(e_chi)
    store['r_mag'].append(r_mag); store['r_chi'].append(r_chi)
    store['kl'].append(kl_fin)

store = {k: np.array(v) for k, v in store.items()}
print(f"\n{'━'*62}")
print("  Entrenamiento completo. Generando figura …")

# =============================================================================
# 6.  Figura — 3 paneles
# =============================================================================
g     = store['g']
BLUE  = '#2563EB'
RED   = '#DC2626'
GREEN = '#16A34A'
G_C   = 1.0

kw_e = dict(color=BLUE, marker='o', ms=6, lw=2.0, label='Exacto')
kw_r = dict(color=RED,  marker='s', ms=6, lw=2.0, ls='--', label='RBM (PCD)')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(
    rf'TFIM 1D  —  $N={N},\;n_h={N_HIDDEN}$,  PCD-{K_CD},  {N_EPOCHS} épocas,  init. PCA',
    fontsize=12
)

def add_gc(ax):
    ax.axvline(G_C, color='#6B7280', ls=':', lw=1.3, label='$g_c = 1$')

# ── Magnetización ─────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(g, store['e_mag'], **kw_e)
ax.plot(g, store['r_mag'], **kw_r)
add_gc(ax)
ax.set(xlabel='$g = |J|/h$',
       ylabel=r'$\langle |m| \rangle$',
       title='Magnetización por sitio',
       ylim=(0, None))
ax.legend(framealpha=0.8)
ax.grid(alpha=0.3)

# ── Susceptibilidad ───────────────────────────────────────────────────────
ax = axes[1]
ax.plot(g, store['e_chi'], **kw_e)
ax.plot(g, store['r_chi'], **kw_r)
add_gc(ax)
ax.set(xlabel='$g = |J|/h$',
       ylabel=r'$\chi = N\,\mathrm{Var}(m)$',
       title='Susceptibilidad magnética',
       ylim=(0, None))
ax.legend(framealpha=0.8)
ax.grid(alpha=0.3)

# ── KL divergencia ────────────────────────────────────────────────────────
ax = axes[2]
ax.semilogy(g, store['kl'], color=GREEN, marker='D', ms=6, lw=2.0,
            label='KL (PCD + init. PCA)')
add_gc(ax)
ax.set(xlabel='$g = |J|/h$',
       ylabel=r'$\mathrm{KL}(p_{\rm data} \| p_{\rm RBM})$',
       title='Divergencia KL')
ax.legend(framealpha=0.8)
ax.grid(alpha=0.3, which='both')

fig.tight_layout()
fig.savefig('tfim_observables_v3.png', dpi=150, bbox_inches='tight')
print("  → tfim_observables_v3.png")
plt.show()
print("\nHecho.")
