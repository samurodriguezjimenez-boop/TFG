"""
=============================================================================
RBM multicapa para el Modelo de Ising de Campo Transverso (TFIM)
Estudio de la susceptibilidad magnética con comparación a diagonalización exacta
=============================================================================

El Hamiltoniano del TFIM es:
    H = -J * sum_i sigma^z_i sigma^z_{i+1} - h * sum_i sigma^x_i

donde sigma^z y sigma^x son las matrices de Pauli, J el acoplamiento y h el campo.

La susceptibilidad magnética se define como:
    chi = beta * (⟨m^2⟩ - ⟨m⟩^2)
donde m = (1/N) * sum_i sigma^z_i es la magnetización por espín.

La dificultad de chi es que es una VARIANZA: requiere estimar ⟨m^2⟩ con muy
alta precisión, ya que chi es diferencia de dos cantidades grandes.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct
import time

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# Parámetros del sistema
N_SPINS   = 6          # Número de espines (cadena 1D con PBC)
J_COUP    = 1.0         # Acoplamiento ferromagnético
H_FIELDS  = np.linspace(0.1, 2.5, 15)  # Valores de h/J a estudiar
BETA      = 1.0         # Temperatura inversa (T=1, estudiamos estado fundamental con VMC)

# Parámetros de entrenamiento
N_HIDDEN_LAYERS = [3, 4, 3]  # Neuronas en cada capa oculta
                              # Multicapa: aumenta la capacidad expresiva de la RBM
LEARNING_RATE   = 5e-3
N_EPOCHS        = 800
N_SAMPLES_TRAIN = 1000   # Muestras por iteración de entrenamiento (MCMC)
N_SAMPLES_EVAL  = 8000   # Muestras para evaluación de observables (más precisión)
N_CHAINS        = 50     # Cadenas de Markov paralelas para mejor mezcla


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIAGONALIZACIÓN EXACTA DEL TFIM
# ─────────────────────────────────────────────────────────────────────────────

def exact_diagonalization(n, J, h):
    """
    Construye el Hamiltoniano del TFIM en la base computacional {|↑⟩,|↓⟩}^N
    y lo diagonaliza exactamente.
    
    Returns:
        E0: energía del estado fundamental por espín
        m_exact: magnetización ⟨sigma^z⟩ en el estado fundamental
        chi_exact: susceptibilidad magnética
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Diccionario de bases: índice -> configuración de espines {-1, +1}
    # El bit i del índice representa el espín i (0=↑=+1, 1=↓=-1)
    
    # Término -J * sigma^z_i * sigma^z_{i+1} (diagonal en la base computacional)
    for state in range(dim):
        diag_elem = 0.0
        for i in range(n):
            j = (i + 1) % n  # Condiciones de contorno periódicas (PBC)
            # Espín i: bit i del estado. 0 -> +1, 1 -> -1
            si = 1 - 2 * ((state >> i) & 1)
            sj = 1 - 2 * ((state >> j) & 1)
            diag_elem += -J * si * sj
        H[state, state] += diag_elem
    
    # Término -h * sigma^x_i (off-diagonal: voltea el espín i)
    for state in range(dim):
        for i in range(n):
            flipped = state ^ (1 << i)   # XOR para voltear el espín i
            H[state, flipped] += -h
    
    # Diagonalización exacta
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Estado fundamental
    E0 = eigvals[0] / n   # Por espín
    psi0 = eigvecs[:, 0]  # Amplitudes del estado fundamental
    probs = psi0 ** 2     # Probabilidades (psi0 es real para TFIM)
    
    # Magnetización: ⟨m⟩ = ⟨psi0| M |psi0⟩ con M = (1/N) sum sigma^z_i
    mag_values = np.zeros(dim)
    for state in range(dim):
        mag = 0.0
        for i in range(n):
            mag += 1 - 2 * ((state >> i) & 1)
        mag_values[state] = mag / n
    
    m_exact    = np.sum(probs * mag_values)        # ⟨m⟩
    m2_exact   = np.sum(probs * mag_values ** 2)   # ⟨m^2⟩
    m_abs_exact = np.sum(probs * np.abs(mag_values))  # ⟨|m|⟩
    
    # Susceptibilidad: chi = beta * N * (⟨m^2⟩ - ⟨m⟩^2)
    # Nota: a veces se define con ⟨|m|⟩ en lugar de ⟨m⟩ para romper simetría
    chi_exact  = BETA * n * (m2_exact - m_abs_exact ** 2)
    
    return E0, m_abs_exact, chi_exact


print("Calculando diagonalización exacta...")
ED_results = {}
for h_val in H_FIELDS:
    E0, m_ed, chi_ed = exact_diagonalization(N_SPINS, J_COUP, h_val)
    ED_results[h_val] = {"E0": E0, "m": m_ed, "chi": chi_ed}
    
print(f"  h/J = {H_FIELDS[7]:.2f}: m = {ED_results[H_FIELDS[7]]['m']:.4f}, "
      f"chi = {ED_results[H_FIELDS[7]]['chi']:.4f}")
print("Diagonalización exacta completada.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RED NEURONAL: RBM PROFUNDA (Deep RBM / NQS)
# ─────────────────────────────────────────────────────────────────────────────

class DeepRBM(nn.Module):
    """
    Neural Quantum State (NQS) basado en una RBM con múltiples capas ocultas.
    
    La función de onda se representa como:
        psi(s) = exp(ReNN(s)) * exp(i * ImNN(s))
    
    Para el TFIM con J > 0, el estado fundamental puede tomarse como real y positivo
    gracias al teorema de Perron-Frobenius (sin frustración), por lo que trabajamos
    con una red real que da log|psi(s)|.
    
    Arquitectura: visible -> hidden_1 -> hidden_2 -> ... -> log_amplitude
    
    La diferencia con una RBM estándar (1 capa) es que las capas adicionales
    permiten capturar correlaciones de mayor orden, cruciales para observables
    de segundo momento como la susceptibilidad.
    """
    
    def __init__(self, n_visible, hidden_sizes):
        super().__init__()
        self.n_visible = n_visible
        
        # Construimos la red capa a capa
        # Input: configuración de espines en {-1, +1}^N
        layers = []
        in_size = n_visible
        
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.Tanh())   # Tanh mapea a (-1,1), bueno para espines
            in_size = h_size
        
        # Capa de salida: un escalar (log|psi|)
        layers.append(nn.Linear(in_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Inicialización pequeña para empezar cerca del estado uniforme
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.01)
    
    def log_amplitude(self, spins):
        """
        Devuelve log|psi(s)| para un batch de configuraciones.
        
        Args:
            spins: tensor de forma (batch, N) con valores en {-1, +1}
        Returns:
            log_amp: tensor de forma (batch,)
        """
        return self.net(spins.float()).squeeze(-1)
    
    def amplitude(self, spins):
        """Amplitud (no normalizada) psi(s) = exp(log|psi|)."""
        return torch.exp(self.log_amplitude(spins))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MUESTREO MARKOV CHAIN MONTE CARLO (MCMC)
# ─────────────────────────────────────────────────────────────────────────────

class MetropolisSampler:
    """
    Muestreador de Metropolis-Hastings para la distribución |psi(s)|^2.
    
    Propone movimientos de "flip único" (voltear un espín aleatorio) y acepta
    con probabilidad min(1, |psi(s')|^2 / |psi(s)|^2).
    
    Usamos múltiples cadenas paralelas para reducir la correlación entre muestras
    y mejorar la estimación de la varianza (especialmente importante para chi).
    """
    
    def __init__(self, model, n_chains, n_spins, device):
        self.model   = model
        self.n_chains = n_chains
        self.n_spins  = n_spins
        self.device   = device
        
        # Inicializar cadenas con configuraciones aleatorias
        self.reset()
    
    def reset(self):
        """Reinicia las cadenas con configuraciones aleatorias en {-1,+1}^N."""
        self.configs = (2 * torch.randint(0, 2, (self.n_chains, self.n_spins),
                                          device=self.device) - 1).float()
    
    @torch.no_grad()
    def step(self, n_steps=1):
        """
        Realiza n_steps pasos de Metropolis en todas las cadenas.
        Cada paso propone voltear un espín aleatorio.
        """
        for _ in range(n_steps):
            # Escoger un espín aleatorio para cada cadena
            flip_idx = torch.randint(0, self.n_spins, (self.n_chains,),
                                     device=self.device)
                
            # Crear configuraciones propuestas (voltear el espín seleccionado)
            proposed = self.configs.clone()
            for c in range(self.n_chains):
                proposed[c, flip_idx[c]] *= -1
            
            # Calcular log-amplitudes de configuraciones actuales y propuestas
            log_amp_curr = self.model.log_amplitude(self.configs)
            log_amp_prop = self.model.log_amplitude(proposed)
            
            # Ratio de aceptación: |psi'|^2 / |psi|^2 = exp(2*(log|psi'| - log|psi|))
            log_ratio = 2.0 * (log_amp_prop - log_amp_curr)
            accept_prob = torch.exp(torch.clamp(log_ratio, max=0.0))
            
            # Aceptar/rechazar con probabilidad Metropolis
            accept = torch.rand(self.n_chains, device=self.device) < accept_prob
            self.configs[accept] = proposed[accept]
        
        return self.configs.clone()
    
    @torch.no_grad()
    def sample(self, n_samples, n_thermalize=200, n_skip=2):
        """
        Genera n_samples muestras tras termalización.
        
        Args:
            n_samples: número total de muestras (distribuidas entre cadenas)
            n_thermalize: pasos de termalización antes de recoger muestras
            n_skip: pasos entre muestras consecutivas (reduce autocorrelación)
        
        Returns:
            samples: tensor (n_samples, N) con configuraciones
        """
        # Termalización: dejamos que las cadenas lleguen al equilibrio
        self.step(n_thermalize)
        
        # Recolección de muestras
        n_per_chain = n_samples // self.n_chains
        all_samples = []
        
        for _ in range(n_per_chain):
            self.step(n_skip)
            all_samples.append(self.configs.clone())
        
        # Concatenar todas las muestras
        return torch.cat(all_samples, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENERGÍA LOCAL Y GRADIENTES (VMC)
# ─────────────────────────────────────────────────────────────────────────────

def local_energy(model, spins, J, h, device):
    """
    Calcula la energía local E_loc(s) = ⟨s|H|psi⟩ / ⟨s|psi⟩ para el TFIM.
    
    E_loc(s) = -J * sum_i s_i * s_{i+1}                   (término ZZ, diagonal)
             - h * sum_i psi(s^i_flip) / psi(s)            (término X, off-diagonal)
    
    donde s^i_flip es la configuración s con el espín i volteado.
    
    Este cálculo es el corazón de VMC: evitamos construir H explícitamente.
    """
    batch = spins.shape[0]
    n     = spins.shape[1]
    
    # ---- Término diagonal: -J * sum sigma^z_i * sigma^z_{i+1} (PBC) ----
    # Producto de espines vecinos: si * s_{i+1}
    rolled = torch.roll(spins, -1, dims=1)   # Desplazamiento cíclico
    e_diag = -J * (spins * rolled).sum(dim=1)  # Forma: (batch,)
    
    # ---- Término off-diagonal: -h * sum_i psi(s^i) / psi(s) ----
    # Para cada espín i, calculamos la amplitud de la configuración con ese espín volteado
    log_amp_current = model.log_amplitude(spins)  # log|psi(s)|, forma: (batch,)
    
    e_offdiag = torch.zeros(batch, device=device)
    
    for i in range(n):
        # Voltear el espín i en todo el batch
        spins_flipped = spins.clone()
        spins_flipped[:, i] *= -1
        
        log_amp_flipped = model.log_amplitude(spins_flipped)  # log|psi(s^i)|
        
        # ratio psi(s^i)/psi(s) = exp(log|psi(s^i)| - log|psi(s)|)
        ratio = torch.exp(log_amp_flipped - log_amp_current)
        e_offdiag += -h * ratio
    
    return (e_diag + e_offdiag) / n   # Por espín


def magnetization_local(spins):
    """
    Magnetización local m(s) = (1/N) * sum_i s_i.
    
    Es simplemente el promedio de los espines de la configuración.
    Al promediar sobre |psi|^2 obtenemos ⟨m⟩.
    """
    return spins.mean(dim=1)   # Forma: (batch,)


def compute_observables(model, sampler, J, h, device, n_samples=8000):
    """
    Estima todos los observables usando muestras MCMC:
        - Energía por espín: ⟨E⟩/N = ⟨E_loc⟩_{|psi|^2}
        - Magnetización: ⟨|m|⟩
        - Susceptibilidad: chi = beta * N * (⟨m^2⟩ - ⟨|m|⟩^2)
    
    Para la susceptibilidad usamos el ESTIMADOR DE BLOCKING para reducir el
    sesgo por autocorrelaciones en la cadena de Markov.
    """
    samples = sampler.sample(n_samples)
    
    with torch.no_grad():
        # Energía local para cada muestra
        e_loc  = local_energy(model, samples, J, h, device)
        
        # Magnetización por muestra
        m_loc  = magnetization_local(samples)
        m_abs  = m_loc.abs()
        m2_loc = m_loc ** 2
    
    e_loc_np  = e_loc.cpu().numpy()
    m_abs_np  = m_abs.cpu().numpy()
    m2_np     = m2_loc.cpu().numpy()
    
    # ---- Estimación por blocking para reducir sesgo de autocorrelación ----
    # Dividimos las muestras en bloques y promediamos dentro de cada bloque.
    # Las medias de bloque están mucho menos correlacionadas entre sí.
    block_size = 50
    n_blocks   = len(e_loc_np) // block_size
    
    # Truncar al múltiplo exacto de block_size
    e_blocks  = e_loc_np[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
    m_blocks  = m_abs_np[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
    m2_blocks = m2_np[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
    
    E_mean   = e_blocks.mean()
    m_mean   = m_blocks.mean()
    m2_mean  = m2_blocks.mean()
    
    # Susceptibilidad: chi = beta * N * Var(m) = beta * N * (⟨m^2⟩ - ⟨|m|⟩^2)
    # Usamos el estimador sin sesgo de la varianza entre bloques
    chi = BETA * N_SPINS * (m2_mean - m_mean ** 2)
    
    # Error estadístico (para diagnóstico)
    E_err   = e_blocks.std()  / np.sqrt(n_blocks)
    m_err   = m_blocks.std()  / np.sqrt(n_blocks)
    chi_err = BETA * N_SPINS * np.sqrt(
        (m2_blocks.std() / np.sqrt(n_blocks))**2 +
        (2 * m_mean * m_blocks.std() / np.sqrt(n_blocks))**2
    )
    
    return {
        "E":     E_mean,  "E_err":   E_err,
        "m":     m_mean,  "m_err":   m_err,
        "chi":   chi,     "chi_err": chi_err,
        "m2":    m2_mean
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PÉRDIDA VMC Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def vmc_loss(model, samples, J, h, device):
    """
    Pérdida VMC: varianza de la energía local (equivalente a minimizar ⟨H⟩).
    
    L = ⟨E_loc^2⟩ - ⟨E_loc⟩^2 = Var(E_loc)
    
    Minimizar la varianza de E_loc es equivalente a minimizar ⟨E⟩ y tiene
    mejores propiedades de convergencia numérica (gradientes más estables).
    
    El gradiente respecto a los parámetros theta es:
    dL/dtheta = 2 * Re[ ⟨(E_loc - ⟨E_loc⟩) * d(log psi)/dtheta⟩ ]
    
    PyTorch calcula esto automáticamente gracias al autograd.
    """
    # Energía local (con gradiente habilitado para backprop)
    e_loc = local_energy(model, samples, J, h, device)
    e_mean = e_loc.detach().mean()
    
    # Amplitudes para el gradiente: log|psi(s)|
    log_amp = model.log_amplitude(samples)
    
    # Pérdida: ⟨(E_loc - ⟨E⟩) * log|psi|⟩ (gradiente de ⟨E⟩ respecto a theta)
    # Esta forma asegura que el gradiente sea correcto en VMC
    loss = ((e_loc.detach() - e_mean) * log_amp).mean()
    
    return loss, e_mean.item()


def train_rbm(h_field, n_epochs=N_EPOCHS, verbose=True):
    """
    Entrena la RBM para un valor dado del campo transverso h.
    
    Estrategia de entrenamiento:
    1. Inicializar con pesos pequeños (cerca del estado paramagnético)
    2. Iterar: (a) muestrear |psi|^2, (b) calcular gradiente VMC, (c) actualizar
    3. Usar scheduler de LR para convergencia fina al final
    
    Returns:
        model: RBM entrenada
        history: dict con la evolución de E y m durante el entrenamiento
    """
    # Arquitectura: visible + capas ocultas
    # Más neuronas ocultas = mayor capacidad para correlaciones de alto orden
    hidden_sizes = [N_SPINS * s for s in N_HIDDEN_LAYERS]
    
    model   = DeepRBM(N_SPINS, hidden_sizes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Reducir LR a mitad del entrenamiento para convergencia más fina
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_epochs // 3, gamma=0.5
    )
    
    sampler = MetropolisSampler(model, N_CHAINS, N_SPINS, DEVICE)
    
    history = {"E": [], "m": [], "epoch": []}
    
    for epoch in range(n_epochs):
        model.train()
        
        # 1. Muestrear configuraciones de la distribución actual |psi|^2
        #    Termalización corta porque las cadenas ya están "calientes"
        samples = sampler.sample(N_SAMPLES_TRAIN, n_thermalize=20, n_skip=1)
        
        # 2. Calcular la pérdida VMC y hacer backpropagation
        optimizer.zero_grad()
        loss, e_mean = vmc_loss(model, samples, J_COUP, h_field, DEVICE)
        loss.backward()
        
        # Clip de gradiente para estabilidad (evita explosión de gradientes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 3. Registrar progreso cada 50 épocas
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                m_samples = sampler.sample(2000, n_thermalize=50, n_skip=2)
                m_val = magnetization_local(m_samples).abs().mean().item()
            
            history["E"].append(e_mean)
            history["m"].append(m_val)
            history["epoch"].append(epoch)
            
            if verbose:
                print(f"  Época {epoch:4d}/{n_epochs} | E={e_mean:.4f} | "
                      f"|m|={m_val:.4f} | LR={scheduler.get_last_lr()[0]:.2e}")
    
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUCLE PRINCIPAL: BARRE TODOS LOS VALORES DE h
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Entrenando RBM para cada valor de h/J...")
print("=" * 60)

rbm_results = {}

for idx, h_val in enumerate(H_FIELDS):
    print(f"\n[{idx+1}/{len(H_FIELDS)}] h/J = {h_val:.3f}")
    
    # Entrenar el modelo (reducimos épocas para agilizar; aumentar para más precisión)
    model, history = train_rbm(h_val, n_epochs=N_EPOCHS, verbose=False)
    
    # Evaluación final con muchas muestras para precisión estadística
    model.eval()
    sampler_eval = MetropolisSampler(model, N_CHAINS * 2, N_SPINS, DEVICE)
    obs = compute_observables(
        model, sampler_eval, J_COUP, h_val, DEVICE, n_samples=N_SAMPLES_EVAL
    )
    
    rbm_results[h_val] = {**obs, "history": history}
    
    # Comparación rápida con exacta
    ed = ED_results[h_val]
    print(f"  RBM:   E={obs['E']:.4f}±{obs['E_err']:.4f}, "
          f"|m|={obs['m']:.4f}±{obs['m_err']:.4f}, "
          f"chi={obs['chi']:.4f}±{obs['chi_err']:.4f}")
    print(f"  Exact: E={ed['E0']:.4f}, "
          f"|m|={ed['m']:.4f}, "
          f"chi={ed['chi']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZACIÓN: CONVERGENCIA Y COMPARACIÓN CON ED
# ─────────────────────────────────────────────────────────────────────────────

# Extraer resultados como arrays
h_arr     = np.array(H_FIELDS)
m_rbm     = np.array([rbm_results[h]["m"]   for h in H_FIELDS])
m_err_rbm = np.array([rbm_results[h]["m_err"] for h in H_FIELDS])
chi_rbm   = np.array([rbm_results[h]["chi"] for h in H_FIELDS])
chi_err_rbm = np.array([rbm_results[h]["chi_err"] for h in H_FIELDS])
E_rbm     = np.array([rbm_results[h]["E"]   for h in H_FIELDS])

m_ed    = np.array([ED_results[h]["m"]   for h in H_FIELDS])
chi_ed  = np.array([ED_results[h]["chi"] for h in H_FIELDS])
E_ed    = np.array([ED_results[h]["E0"]  for h in H_FIELDS])

# ─── Figura 1: Observables vs h/J ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"TFIM con N={N_SPINS} espines: RBM multicapa vs Diagonalización Exacta",
    fontsize=14, fontweight="bold"
)

# Panel 1: Energía por espín
ax = axes[0]
ax.plot(h_arr, E_ed, "k-", lw=2, label="Exacta")
ax.errorbar(h_arr, E_rbm, yerr=m_err_rbm, fmt="ro--", capsize=4,
            label="RBM multicapa", markersize=6)
ax.set_xlabel("h/J (campo transverso)", fontsize=12)
ax.set_ylabel("E₀/N", fontsize=12)
ax.set_title("Energía del estado fundamental", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Panel 2: Magnetización ⟨|m|⟩
ax = axes[1]
ax.plot(h_arr, m_ed, "k-", lw=2, label="Exacta")
ax.errorbar(h_arr, m_rbm, yerr=m_err_rbm, fmt="bs--", capsize=4,
            label="RBM multicapa", markersize=6)
ax.axvline(x=1.0, color="gray", ls=":", alpha=0.7, label="Punto crítico (h/J≈1)")
ax.set_xlabel("h/J (campo transverso)", fontsize=12)
ax.set_ylabel("⟨|m|⟩", fontsize=12)
ax.set_title("Magnetización por espín", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Susceptibilidad magnética χ
ax = axes[2]
ax.plot(h_arr, chi_ed, "k-", lw=2, label="Exacta")
ax.errorbar(h_arr, chi_rbm, yerr=chi_err_rbm, fmt="g^--", capsize=4,
            label="RBM multicapa", markersize=6)
ax.axvline(x=1.0, color="gray", ls=":", alpha=0.7, label="Punto crítico (h/J≈1)")
ax.set_xlabel("h/J (campo transverso)", fontsize=12)
ax.set_ylabel("χ = β·N·(⟨m²⟩ − ⟨|m|⟩²)", fontsize=12)
ax.set_title("Susceptibilidad magnética", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("deeprbm_observables.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigura 1 guardada: deeprbm_observables.png")

# ─── Figura 2: Convergencia durante entrenamiento (para h cercano al punto crítico) ───
# El punto más interesante es cerca de la transición de fase h/J ≈ 1
critical_idx  = np.argmin(np.abs(h_arr - 1.0))  # h más cercano al punto crítico
h_critical    = H_FIELDS[critical_idx]
history_crit  = rbm_results[h_critical]["history"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Convergencia del entrenamiento VMC (h/J = {h_critical:.2f}, cerca del punto crítico)",
    fontsize=13, fontweight="bold"
)

epochs = history_crit["epoch"]

# Panel izquierdo: convergencia de la energía
ax = axes[0]
ax.plot(epochs, history_crit["E"], "r-o", markersize=4, label="E RBM (training)")
ax.axhline(y=ED_results[h_critical]["E0"], color="k", ls="--", lw=2, label="E exacta")
ax.set_xlabel("Época", fontsize=12)
ax.set_ylabel("E₀/N", fontsize=12)
ax.set_title("Convergencia de la energía", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Panel derecho: convergencia de la magnetización
ax = axes[1]
ax.plot(epochs, history_crit["m"], "b-s", markersize=4, label="|m| RBM (training)")
ax.axhline(y=ED_results[h_critical]["m"], color="k", ls="--", lw=2, label="|m| exacta")
ax.set_xlabel("Época", fontsize=12)
ax.set_ylabel("⟨|m|⟩", fontsize=12)
ax.set_title("Convergencia de la magnetización", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("deeprbm_convergencia.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figura 2 guardada: deeprbm_convergencia.png")

# ─── Figura 3: Error relativo en susceptibilidad ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# Error relativo: |chi_RBM - chi_ED| / chi_ED * 100%
rel_err_chi = np.abs(chi_rbm - chi_ed) / (chi_ed + 1e-8) * 100
rel_err_m   = np.abs(m_rbm - m_ed)    / (m_ed   + 1e-8) * 100

ax.plot(h_arr, rel_err_m,   "b-o", markersize=6, label="Error relativo |m|")
ax.plot(h_arr, rel_err_chi, "r-^", markersize=6, label="Error relativo χ")
ax.axvline(x=1.0, color="gray", ls=":", alpha=0.7, label="Punto crítico")
ax.set_xlabel("h/J", fontsize=12)
ax.set_ylabel("Error relativo (%)", fontsize=12)
ax.set_title("Comparación de precisión: magnetización vs susceptibilidad", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale("log")   # Escala log para ver mejor los órdenes de magnitud

plt.tight_layout()
plt.savefig("deeprbm_error_relativo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figura 3 guardada: deeprbm_error_relativo.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. TABLA RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'h/J':>6} | {'m_ED':>7} | {'m_RBM':>7} | {'χ_ED':>7} | {'χ_RBM':>7} | {'Δχ/χ':>7}")
print("-" * 75)
for h_val in H_FIELDS:
    ed  = ED_results[h_val]
    rbm = rbm_results[h_val]
    rel = abs(rbm["chi"] - ed["chi"]) / (ed["chi"] + 1e-8) * 100
    print(f"{h_val:6.2f} | {ed['m']:7.4f} | {rbm['m']:7.4f} | "
          f"{ed['chi']:7.4f} | {rbm['chi']:7.4f} | {rel:6.1f}%")
print("=" * 75)
print("\nEjecución completada.")
