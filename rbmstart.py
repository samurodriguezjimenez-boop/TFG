# Lets simulate the 1D quantum Ising model using a simple RBM
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import jax.numpy as jnp
from jax import random as jax_random
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube

# Parameters
N = 6  # Number of spins
J = 100.0  # Interaction strength
h = -1.0 # Transverse field strength

# Initialize Hilbert space and Hamiltonian
hi = Spin(s=1 / 2, N=N)
g = Hypercube(length=N, n_dim=1, pbc=True)
H = Ising(hilbert=hi, graph=g, h=h, J=J)

# Initialize a random spin configuration
def random_configuration(N):
    return np.random.choice([-1, 1], size=N)
c0 = random_configuration(N)
print('Configuración inicial: ', c0)

# Calculate the energy of the initial configuration
def config_energy(c, operator):
    """Return the diagonal matrix element <c|operator|c> for a configuration c.

    Uses NetKet's `get_conn_padded` which returns connected configurations
    and matrix elements. For a single configuration we sum the matrix
    elements whose connected configuration equals the original one.
    """
    x = jnp.array(c)[None, ...]
    xp, mels = operator.get_conn_padded(x)
    # xp shape: (1, n_conn, N), mels shape: (1, n_conn)
    eq = jnp.all(xp[0] == x[0], axis=-1)
    return float(jnp.sum(mels[0] * eq))
initial_energy = config_energy(c0, H)

# Lets now programme a simple RBM to learn the ground state of the Hamiltonian
# We will use a simple RBM with binary visible units and binary hidden units. The visible units will represent the spin configuration and the hidden units will represent the correlations between the spins.
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.random.normal(0, 0.01, size=(n_visible, n_hidden))  # Weights
        self.b = np.zeros(n_visible)  # Visible biases
        self.c = np.zeros(n_hidden)   # Hidden biases
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample_h(self, v):
        h_prob = self.sigmoid(np.dot(v, self.W) + self.c)
        return (h_prob > np.random.rand(self.n_hidden)).astype(float)
    
    def sample_v(self, h):
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.b)
        return (v_prob > np.random.rand(self.n_visible)).astype(float)
    
    def energy(self, v):
        h_prob = self.sigmoid(np.dot(v, self.W) + self.c)
        return -np.dot(v, self.b) - np.sum(np.log(1 + np.exp(np.dot(v, self.W) + self.c)))
    
# Now we will train the RBM using contrastive divergence
def contrastive_divergence(rbm, v0, k=1):
    v = v0
    for _ in range(k):
        h = rbm.sample_h(v)
        v = rbm.sample_v(h)
    return v

# Training the RBM
rbm = RBM(n_visible=N, n_hidden=10)
n_epochs = 1000
learning_rate = 0.01
for epoch in range(n_epochs):
    v0 = random_configuration(N)
    vk = contrastive_divergence(rbm, v0)
    
    # Update weights and biases
    h0 = rbm.sample_h(v0)
    hk = rbm.sample_h(vk)
    
    rbm.W += learning_rate * (np.outer(v0, h0) - np.outer(vk, hk))
    rbm.b += learning_rate * (v0 - vk)
    rbm.c += learning_rate * (h0 - hk)

# After training, we can sample from the RBM to get a configuration that approximates the ground state
v_sample = rbm.sample_v(rbm.sample_h(random_configuration(N)))
print('Configuración muestreada por el RBM: ', v_sample)
energy_sample = config_energy(v_sample, H)

# Plot the distribution of configurations sampled by the RBM
# We asign a unique index to each configuration by interpreting the binary vector as a base 10 number
def config_to_index(config):
    return int("".join(str(int(x)) for x in config), 2)
samples = []
for _ in range(1000):
    v_sample = rbm.sample_v(rbm.sample_h(random_configuration(N)))
    samples.append(config_to_index(v_sample))
plt.hist(samples, bins=2**N)
plt.xlabel('Configuration index')
plt.ylabel('Counts')
plt.title('Histogram of configurations sampled by the RBM')
plt.show()
