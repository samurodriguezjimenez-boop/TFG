import numpy as np
from collections import Counter


class RBM:
    """
    Restricted Boltzmann Machine para unidades binarias {0, 1}.

    Convencion de energia libre (h marginalizados):
        F(v) = -v*b - sum_j log(1 + exp((W^T v + c)_j))

    La distribucion del modelo es:
        p(v) = exp(-F(v)) / Z,   Z = sum_v exp(-F(v))
    """

    def __init__(self, n_visible, n_hidden, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.W = np.random.normal(0, 0.01, size=(n_visible, n_hidden))
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

        # Matriz (2^n_visible, n_visible) con todas las configuraciones binarias.
        # Se construye una sola vez; no cambia durante el entrenamiento.
        self._all_configs = self._build_all_configs(n_visible)

    # ------------------------------------------------------------------ #
    #  Utilidades internas                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sigmoid(x):
        """Sigmoid numericamente estable."""
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    @staticmethod
    def _build_all_configs(n_visible):
        """
        Devuelve (2^n_visible, n_visible) con todas las configuraciones
        binarias, usando operaciones de bits vectorizadas.
        """
        n_configs = 2 ** n_visible
        idx  = np.arange(n_configs, dtype=np.uint32)
        bits = np.arange(n_visible - 1, -1, -1, dtype=np.uint32)
        return ((idx[:, None] >> bits[None, :]) & 1).astype(np.float64)

    def _free_energies_batch(self, V):
        """
        Calcula F(v) = -v*b - sum_j softplus(W^T v + c)_j
        para un batch V de shape (M, n_visible).

        Devuelve array (M,).
        """
        pre = V @ self.W + self.c                              # (M, n_hidden)
        # softplus estable: log(1+exp(x)) = x para x>>0
        softplus = np.where(
            pre > 20,
            pre,
            np.log1p(np.exp(np.clip(pre, -500, 20)))
        )
        return -(V @ self.b) - softplus.sum(axis=1)            # (M,)

    def _log_partition(self):
        """
        log Z = log sum_v exp(-F(v)), calculado de forma numericamente estable
        mediante el truco log-sum-exp:

            log Z = -F_min + log( sum_i exp(-(F_i - F_min)) )

        Es decir:
            Z = exp(-F_min) * sum_i exp(-(F_i - F_min))
        """
        F      = self._free_energies_batch(self._all_configs)  # (2^N,)
        F_min  = F.min()
        log_Z  = -F_min + np.log(np.exp(-(F - F_min)).sum())
        return log_Z

    # ------------------------------------------------------------------ #
    #  Muestreo                                                            #
    # ------------------------------------------------------------------ #

    def sample_h(self, v):
        """Muestra binaria h | v. v puede ser (n_visible,) o (M, n_visible)."""
        h_prob = self._sigmoid(v @ self.W + self.c)
        return (h_prob > np.random.rand(*h_prob.shape)).astype(np.float64)

    def sample_v(self, h):
        """Muestra binaria v | h. h puede ser (n_hidden,) o (M, n_hidden)."""
        v_prob = self._sigmoid(h @ self.W.T + self.b)
        return (v_prob > np.random.rand(*v_prob.shape)).astype(np.float64)

    def prob_h_given_v(self, v):
        """P(h=1|v) continua. v puede ser (n_visible,) o (M, n_visible)."""
        return self._sigmoid(v @ self.W + self.c)

    # ------------------------------------------------------------------ #
    #  Energia libre y funcion de particion                                #
    # ------------------------------------------------------------------ #

    def free_energy(self, v):
        """Energia libre para un unico vector v (n_visible,)."""
        return self._free_energies_batch(v[None, :])[0]

    def partition_function(self):
        """Z exacta, vectorizada sobre los 2^n_visible estados."""
        return np.exp(self._log_partition())

    # ------------------------------------------------------------------ #
    #  Divergencia KL                                                      #
    # ------------------------------------------------------------------ #

    def kl_divergence_from_samples(self, dataset):
        """
        KL(p_data || p_model) estimada con frecuencias del dataset.
        Util cuando no se dispone de psi_prob exacto.
        """
        def to_tuple(v):
            return tuple(int(x) for x in np.asarray(v).flatten())

        counts = Counter(to_tuple(v) for v in dataset)
        total  = len(dataset)
        log_Z  = self._log_partition()

        DKL = 0.0
        for v_tuple, count in counts.items():
            v         = np.array(v_tuple, dtype=np.float64)
            p_data    = count / total
            log_p_mod = -self.free_energy(v) - log_Z
            DKL      += p_data * (np.log(p_data) - log_p_mod)
        return float(DKL)

    def kl_divergence_exact(self, psi_prob, hi_states):
        """
        KL(p_data || p_model) usando la distribucion exacta del estado
        fundamental. Totalmente vectorizada.

        Parametros
        ----------
        psi_prob  : array (2^N,)   -- |psi_0(s)|^2 para cada configuracion
        hi_states : array (2^N, N) -- hi.all_states() de NetKet, en {-1,+1}
        """
        p_data     = psi_prob / psi_prob.sum()
        states_bin = (hi_states + 1.0) / 2.0                  # {-1,+1} -> {0,1}

        # Energias libres de todos los estados
        F     = self._free_energies_batch(states_bin)          # (2^N,)

        # log Z calculado sobre _all_configs (base binaria estandar completa)
        log_Z = self._log_partition()

        # log p_model(v) = -F(v) - log Z
        log_p_model = -F - log_Z                               # (2^N,)

        # KL = sum_v p_data(v) * [log p_data(v) - log p_model(v)]
        mask = p_data > 0
        DKL  = np.sum(
            p_data[mask] * (np.log(p_data[mask]) - log_p_model[mask])
        )
        return float(DKL)

    # ------------------------------------------------------------------ #
    #  Contrastive Divergence vectorizada por batch                        #
    # ------------------------------------------------------------------ #

    def contrastive_divergence_batch(self, V0, k=1):
        """
        CD-k para un batch V0 de shape (M, n_visible).
        Devuelve Vk de shape (M, n_visible).
        """
        V = V0.copy()
        for _ in range(k):
            H = self.sample_h(V)
            V = self.sample_v(H)
        return V

    # ------------------------------------------------------------------ #
    #  Entrenamiento                                                        #
    # ------------------------------------------------------------------ #

    def train(self,
              dataset,
              n_epochs      = 1000,
              batch_size    = 100,
              learning_rate = 0.01,
              k             = 5,
              log_every     = 10,
              kl_mode       = 'samples',
              psi_prob      = None,
              hi_states     = None):
        """
        Entrena la RBM con CD-k completamente vectorizado.

        El gradiente se calcula sobre el batch completo de una vez:
            dW = (V0^T H0_prob - Vk^T Hk_prob) / M
            db = mean(V0 - Vk,        axis=0)
            dc = mean(H0_prob-Hk_prob, axis=0)

        Parametros
        ----------
        kl_mode : 'samples' | 'exact'
            'exact' requiere psi_prob y hi_states.

        Devuelve
        --------
        kl_history : list de tuplas (epoch, kl)
        """
        if kl_mode == 'exact' and (psi_prob is None or hi_states is None):
            raise ValueError("kl_mode='exact' requiere psi_prob y hi_states.")

        n_samples       = len(dataset)
        effective_batch = min(batch_size, n_samples)
        kl_history      = []

        for epoch in range(n_epochs):

            # Mini-batch aleatorio
            idx = np.random.choice(n_samples, size=effective_batch,
                                   replace=False)
            V0  = dataset[idx]                                 # (M, n_visible)

            # Fase positiva: estadisticas bajo los datos
            H0_prob = self.prob_h_given_v(V0)                  # (M, n_hidden)

            # Fase negativa: estadisticas bajo el modelo (CD-k)
            Vk      = self.contrastive_divergence_batch(V0, k=k)
            Hk_prob = self.prob_h_given_v(Vk)                  # (M, n_hidden)

            # Gradientes como productos matriciales
            dW = (V0.T @ H0_prob - Vk.T @ Hk_prob) / effective_batch
            db = (V0 - Vk).mean(axis=0)
            dc = (H0_prob - Hk_prob).mean(axis=0)

            # Actualizacion de parametros
            self.W += learning_rate * dW
            self.b += learning_rate * db
            self.c += learning_rate * dc

            # Registro de KL cada log_every epocas
            if epoch % log_every == 0:
                if kl_mode == 'exact':
                    kl = self.kl_divergence_exact(psi_prob, hi_states)
                else:
                    kl = self.kl_divergence_from_samples(dataset)
                kl_history.append((epoch, kl))

        return kl_history
