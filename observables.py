"""
TFIM Observable Study v2 — RBM vs Exact Ground State
=====================================================
Fixes sobre v1
--------------
  1. PCD (Persistent CD) en lugar de CD estándar:
       las cadenas de la fase negativa son persistentes entre mini-batches,
       lo que permite mezclar entre los dos modos ferromagnéticos a lo largo
       del entrenamiento. Esto es crucial en la fase ordenada (g > 1), donde
       la distribución es bimodal (|↑↑...↑⟩ + |↓↓...↓⟩) y CD-k con k pequeño
       nunca cruza la barrera energética.

  2. Correlación conectada C_c(r) = ⟨σᵢσᵢ₊ᵣ⟩ - ⟨σᵢ⟩²:
       El estimador anterior de ξ era inestable porque usaba C(r) bruta.
       - Fase ordenada : C(r) ≈ m² ≠ 0  → log(C) ≈ const, pendiente ≈ 0 → ξ = ∞/NaN
       - Fase desordenada: C(r) ≈ 0       → log(0) = -∞
       La correlación conectada decae correctamente en ambas fases:
         · Desordenada : decaimiento exponencial → ξ finita
         · Ordenada    : plana ≈ 0 → ξ → ∞  (long-range order)
         · Crítica     : decaimiento algebraico → ξ grande pero finita (tamaño finito)

Uso
---
  python tfim_observables_v2.py

Requiere rbm.py en el mismo directorio.
"""

import numpy as np
import netket as nk
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import warnings

from rbmopt import RBM

np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# 0.  Configuración global
# =============================================================================
N          = 14
J          = -1.0
N_HIDDEN   = 2 * N       # 20 unidades ocultas
N_TRAIN    = 5_000
N_OBS      = 10_000      # cadenas Gibbs para observables RBM
N_EPOCHS   = 3_000       # más épocas para que PCD mezcle bien
BATCH_SIZE = 100
LR         = 0.01
K_CD       = 1           # PCD-1 es suficiente; las cadenas son persistentes
LOG_EVERY  = 500
N_WARMUP   = 1_000       # warm-up largo para muestreo final
SEED       = 42

# 13 valores de g — densos cerca de g_c = 1
G_VALUES = np.array([
    0.20, 0.40, 0.60,
    0.75, 0.85, 0.95,
    1.00,
    1.05, 1.15, 1.25,
    1.50, 2.00, 3.00,
])

G_PLOT_CR = [0.40, 1.00, 2.00]   # g para la figura de C_c(r)

# =============================================================================
# 1.  RBM con PCD
# =============================================================================
class RBM_PCD(RBM):
    """
    Extiende RBM con Persistent Contrastive Divergence.

    Diferencia clave respecto a CD:
      CD  : cadenas negativas parten de los datos en cada mini-batch.
      PCD : cadenas negativas se inicializan una vez y continúan entre batches.
            Cada actualización de parámetros desplaza ligeramente la distribución
            del modelo, pero las cadenas siguen explorando y (con suficientes pasos
            acumulados) pueden cruzar barreras energéticas entre modos.
    """

    def train_pcd(self,
                  dataset,
                  n_epochs      = 1000,
                  batch_size    = 100,
                  learning_rate = 0.01,
                  k             = 1,
                  log_every     = 10,
                  kl_mode       = 'samples',
                  psi_prob      = None,
                  hi_states     = None):

        if kl_mode == 'exact' and (psi_prob is None or hi_states is None):
            raise ValueError("kl_mode='exact' requiere psi_prob y hi_states.")

        n_samples       = len(dataset)
        effective_batch = min(batch_size, n_samples)
        rng             = np.random.default_rng(SEED)

        # ── Inicializar cadenas persistentes aleatoriamente ─────────────────
        # Importante: la mitad apuntando hacia arriba y la mitad hacia abajo
        # para evitar modo collapse desde el inicio.
        half = effective_batch // 2
        chains_up   = np.ones((half, self.n_visible), dtype=float)
        chains_down = np.zeros((effective_batch - half, self.n_visible), dtype=float)
        self._pchains = np.vstack([chains_up, chains_down])

        kl_history = []

        for epoch in range(n_epochs):

            # Mini-batch aleatorio
            idx = rng.choice(n_samples, size=effective_batch, replace=False)
            V0  = dataset[idx]                         # (M, n_visible)

            # Fase positiva
            H0_prob = self.prob_h_given_v(V0)          # (M, n_hidden)

            # Fase negativa — k pasos sobre cadenas persistentes
            Vk = self._pchains
            for _ in range(k):
                Hk = self.sample_h(Vk)
                Vk = self.sample_v(Hk)
            self._pchains = Vk.copy()                  # guardar estado de cadenas

            Hk_prob = self.prob_h_given_v(Vk)          # (M, n_hidden)

            # Gradientes
            dW = (V0.T @ H0_prob - Vk.T @ Hk_prob) / effective_batch
            db = (V0 - Vk).mean(axis=0)
            dc = (H0_prob - Hk_prob).mean(axis=0)

            self.W += learning_rate * dW
            self.b += learning_rate * db
            self.c += learning_rate * dc

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
    return (all_states[idx] + 1.0) / 2.0               # {-1,+1} → {0,1}


def gibbs_samples_01(rbm, n_samples, n_warmup=500, seed=None):
    """
    n_samples cadenas paralelas, n_warmup pasos de calentamiento.
    Inicializa la mitad en all-1 y la mitad en all-0 para cubrir ambos modos.
    """
    half    = n_samples // 2
    v_up    = np.ones( (half,           rbm.n_visible), dtype=float)
    v_down  = np.zeros((n_samples-half, rbm.n_visible), dtype=float)
    v       = np.vstack([v_up, v_down])
    for _ in range(n_warmup):
        h = rbm.sample_h(v)
        v = rbm.sample_v(h)
    return v


# =============================================================================
# 4.  Observables con correlación CONECTADA
# =============================================================================
def _connected_corr(states_pm1):
    """
    C_c(r) = ⟨σᵢ σᵢ₊ᵣ⟩ - ⟨σᵢ⟩²

    La correlación conectada elimina el cuadrado de la magnetización media,
    lo que la hace bien comportada tanto en la fase ordenada (donde se vuelve
    plana ≈ 0 indicando long-range order) como en la desordenada (decaimiento
    exponencial neto desde C_c(1) > 0).

    states_pm1 : (M, N) en {-1, +1}
    Devuelve array de longitud N//2 + 1 (r = 0, 1, …, N//2)
    """
    N       = states_pm1.shape[1]
    max_r   = N // 2
    m_site  = states_pm1.mean(axis=0)              # ⟨σᵢ⟩  (N,)
    m2_mean = (m_site**2).mean()                   # ⟨⟨σᵢ⟩²⟩ promediado en i

    Cc = np.zeros(max_r + 1)
    for r in range(max_r + 1):
        corr_full = (states_pm1 * np.roll(states_pm1, -r, axis=1)).mean()
        # Conectada: ⟨σᵢσⱼ⟩ - ⟨σᵢ⟩⟨σⱼ⟩  (promediado sobre pares (i, i+r))
        m_shift   = (m_site * np.roll(m_site, -r)).mean()
        Cc[r]     = corr_full - m_shift
    return Cc


def _connected_corr_exact(all_states, prob):
    """Igual pero con la distribución exacta (vectorizado)."""
    N     = all_states.shape[1]
    max_r = N // 2

    # ⟨σᵢ⟩  para cada sitio  (N,)
    m_site = (all_states * prob[:, None]).sum(axis=0)

    Cc = np.zeros(max_r + 1)
    for r in range(max_r + 1):
        corr_full = np.dot(
            (all_states * np.roll(all_states, -r, axis=1)).mean(axis=1), prob
        )
        m_shift = (m_site * np.roll(m_site, -r)).mean()
        Cc[r]   = corr_full - m_shift
    return Cc


def corr_length(Cc):
    """
    Estima ξ ajustando log(|C_c(r)|) = const - r/ξ  para r ≥ 1.

    - Si todos los |C_c(r)| < umbral (fase desordenada extrema): devuelve NaN
    - Si la pendiente ≥ 0 (fase ordenada: correlaciones no decaen): devuelve np.inf
    - En otro caso: ξ = -1/pendiente
    """
    r_vals = np.arange(1, len(Cc))
    Cv     = np.abs(Cc[1:])
    valid  = Cv > 1e-10
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(r_vals[valid], np.log(Cv[valid]), 1)
    if slope >= 0:
        return np.inf
    return -1.0 / slope


def observables_from_samples(states_pm1):
    """MC: |m|, χ, C_c(r), ξ desde cadenas Gibbs."""
    N   = states_pm1.shape[1]
    m   = states_pm1.mean(axis=1)
    mag = np.abs(m).mean()
    chi = N * (np.mean(m**2) - mag**2)
    Cc  = _connected_corr(states_pm1)
    xi  = corr_length(Cc)
    return mag, chi, Cc, xi


def observables_exact(all_states, prob):
    """Exacto: |m|, χ, C_c(r), ξ desde distribución cuántica."""
    N   = all_states.shape[1]
    m   = all_states.mean(axis=1)
    mag = np.dot(np.abs(m), prob)
    chi = N * (np.dot(m**2, prob) - mag**2)
    Cc  = _connected_corr_exact(all_states, prob)
    xi  = corr_length(Cc)
    return mag, chi, Cc, xi


# =============================================================================
# 5.  Bucle principal
# =============================================================================
store = {k: [] for k in
         ('g', 'e_mag', 'e_chi', 'e_xi',
                'r_mag', 'r_chi', 'r_xi', 'kl')}
cr_store = {}

print(f"\n{'━'*62}")
print(f"  TFIM Observable Study v2 — PCD + Correlación conectada")
print(f"  N={N}   n_h={N_HIDDEN}   {N_EPOCHS} épocas   PCD-{K_CD}")
print(f"{'━'*62}")

for g in G_VALUES:
    h = abs(J) / g
    print(f"\n  g = {g:.3f}   (h = {h:.4f})")
    print(f"  {'─'*42}")

    # ── Exacto ─────────────────────────────────────────────────────────────
    prob, all_states = exact_ground_state(N, h, J)
    e_mag, e_chi, e_Cc, e_xi = observables_exact(all_states, prob)
    xi_str = f"{e_xi:.4f}" if np.isfinite(e_xi) else "∞"
    print(f"  Exact → |m|={e_mag:.4f}  χ={e_chi:.4f}  ξ={xi_str}")

    # ── Entrenamiento PCD ──────────────────────────────────────────────────
    dataset = make_dataset_01(all_states, prob, N_TRAIN, seed=SEED)
    rbm     = RBM_PCD(n_visible=N, n_hidden=N_HIDDEN, seed=SEED)

    kl_hist = rbm.train_pcd(
        dataset,
        n_epochs      = N_EPOCHS,
        batch_size    = BATCH_SIZE,
        learning_rate = LR,
        k             = K_CD,
        log_every     = LOG_EVERY,
        kl_mode       = 'exact',
        psi_prob      = prob,
        hi_states     = all_states,
    )
    kl_0   = kl_hist[0][1]
    kl_fin = kl_hist[-1][1]
    print(f"  KL   → {kl_0:.5f} → {kl_fin:.5f}  "
          f"({'↓' if kl_fin < kl_0 else '↑'} entrenamiento)")

    # ── Muestreo Gibbs y observables RBM ──────────────────────────────────
    rbm_01  = gibbs_samples_01(rbm, N_OBS, N_WARMUP, seed=SEED)
    rbm_pm1 = 2.0 * rbm_01 - 1.0
    r_mag, r_chi, r_Cc, r_xi = observables_from_samples(rbm_pm1)
    xi_str_r = f"{r_xi:.4f}" if np.isfinite(r_xi) else "∞"
    print(f"  RBM  → |m|={r_mag:.4f}  χ={r_chi:.4f}  ξ={xi_str_r}")

    # ── Acumular ───────────────────────────────────────────────────────────
    store['g'].append(g)
    store['e_mag'].append(e_mag); store['e_chi'].append(e_chi)
    store['e_xi'].append(e_xi)
    store['r_mag'].append(r_mag); store['r_chi'].append(r_chi)
    store['r_xi'].append(r_xi);   store['kl'].append(kl_fin)

    if any(abs(g - gp) < 0.02 for gp in G_PLOT_CR):
        cr_store[g] = (e_Cc, r_Cc)

store = {k: np.array(v) for k, v in store.items()}
print(f"\n{'━'*62}")
print("  Entrenamiento completo. Generando figuras …")

# =============================================================================
# 6.  Figuras
# =============================================================================
g    = store['g']
BLUE  = '#2563EB'
RED   = '#DC2626'
GREEN = '#16A34A'
G_C   = 1.0

kw_e  = dict(color=BLUE, marker='o', ms=6, lw=2.0, label='Exact')
kw_r  = dict(color=RED,  marker='s', ms=6, lw=2.0, ls='--', label='RBM (PCD)')

def add_gc(ax):
    ax.axvline(G_C, color='#6B7280', ls=':', lw=1.3, label='$g_c = 1$')


# ── Figura 1: Observables principales ──────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
fig1.suptitle(
    rf'TFIM 1D — Observables ($N={N},\;n_h={N_HIDDEN}$, PCD-{K_CD}, {N_EPOCHS} épocas)',
    fontsize=12, y=0.998
)

# Magnetización
ax = axes[0, 0]
ax.plot(g, store['e_mag'], **kw_e)
ax.plot(g, store['r_mag'], **kw_r)
add_gc(ax)
ax.set(xlabel='$g = |J|/h$', ylabel=r'$\langle |m| \rangle$',
       title='Magnetización por sitio')
ax.legend(framealpha=0.8); ax.grid(alpha=0.3)

# Susceptibilidad
ax = axes[0, 1]
ax.plot(g, store['e_chi'], **kw_e)
ax.plot(g, store['r_chi'], **kw_r)
add_gc(ax)
ax.set(xlabel='$g = |J|/h$', ylabel=r'$\chi = N\,\mathrm{Var}(m)$',
       title='Susceptibilidad')
ax.legend(framealpha=0.8); ax.grid(alpha=0.3)



# Longitud de correlación (infinitos → NaN para plotting limpio)
e_xi_pl = np.where(np.isinf(store['e_xi']), np.nan, store['e_xi'])
r_xi_pl = np.where(np.isinf(store['r_xi']), np.nan, store['r_xi'])

ax = axes[1, 0]
ax.plot(g, e_xi_pl, **kw_e)
ax.plot(g, r_xi_pl, **kw_r)
# Marcar long-range order (ξ = ∞) con flechas en la parte superior
y_top = np.nanmax(np.concatenate([e_xi_pl[~np.isnan(e_xi_pl)],
                                   r_xi_pl[~np.isnan(r_xi_pl)],
                                   [1.0]])) * 1.15
for arr, col in [(store['e_xi'], BLUE), (store['r_xi'], RED)]:
    for gv, xi in zip(g, arr):
        if np.isinf(xi):
            ax.annotate('∞', xy=(gv, y_top * 0.97),
                        ha='center', va='top', fontsize=9,
                        color=col, fontweight='bold')
ax.set_ylim(bottom=0, top=y_top * 1.05)
add_gc(ax)
ax.set(xlabel='$g = |J|/h$', ylabel=r'$\xi$',
       title='Longitud de correlación (conectada)')
ax.legend(framealpha=0.8); ax.grid(alpha=0.3)


# KL final
ax = axes[1, 1]
ax.semilogy(g, store['kl'], color=GREEN, marker='D', ms=6, lw=2.0,
            label='KL (PCD)')
add_gc(ax)
ax.set(xlabel='$g = |J|/h$',
       ylabel=r'$\mathrm{KL}(p_{\rm data} \| p_{\rm RBM})$',
       title='Divergencia KL (final del entrenamiento)')
ax.legend(framealpha=0.8); ax.grid(alpha=0.3, which='both')

fig1.tight_layout()
fig1.savefig('tfim_observables_v2.png', dpi=150, bbox_inches='tight')
print("  → tfim_observables_v2.png")

# ── Figura 2: C_c(r) para g seleccionados ──────────────────────────────────
n_cr  = len(cr_store)
fig2, axes2 = plt.subplots(1, n_cr, figsize=(5 * n_cr, 4.2), sharey=False)
if n_cr == 1:
    axes2 = [axes2]
fig2.suptitle(
    rf'Correlación conectada $C_c(r) = \langle\sigma_i\sigma_{{i+r}}\rangle - \langle\sigma_i\rangle^2$'
    rf'  —  TFIM 1D ($N={N}$)',
    fontsize=11
)

phases = {0.40: 'fase desordenada', 1.00: 'punto crítico', 2.00: 'fase ordenada'}
for ax, (gp, (e_Cc, r_Cc)) in zip(axes2, sorted(cr_store.items())):
    r = np.arange(len(e_Cc))
    ax.plot(r, e_Cc, **kw_e)
    ax.plot(r, r_Cc, **kw_r)
    phase = next((v for k, v in phases.items() if abs(gp - k) < 0.15), '')
    ax.set(xlabel='$r$',
           ylabel=r'$C_c(r)$',
           title=rf'$g = {gp:.2f}$' + (f'\n({phase})' if phase else ''))
    ax.legend(framealpha=0.8); ax.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig('tfim_correlations_v2.png', dpi=150, bbox_inches='tight')
print("  → tfim_correlations_v2.png")

plt.show()



print("\nHecho.")

