#We are going to build a simple MCMC sampler using the Metropolis-Hastings algorithm for the 1D Ising model TFIM.
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
J = 0.5  # Interaction strength
h = -1.0 # Transverse field strength

# Initialize Hilbert space and Hamiltonian

g = Hypercube(length=N, n_dim=1, pbc=True)
hi = Spin(s=1 / 2, N=g.n_nodes)
H = Ising(hilbert=hi, graph=g, h=h, J=J)

# Initialize a random spin configuration
def random_configuration(N):
    return np.random.choice([-1, 1], size=N)
c0 = random_configuration(N)

# Metropolis-Hastings MCMC sampler

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

def metropolis_hastings(c0, H, n_steps, beta):
    c = c0.copy()
    samples = []
    for step in range(n_steps):
        # Propose a new configuration by flipping a random spin
        i = random.randint(0, N-1)
        c_new = c.copy()
        c_new[i] *= -1
        
        # Calculate energy difference
        E_old = config_energy(c, H)
        E_new = config_energy(c_new, H)
        delta_E = E_new - E_old

        
        # Metropolis acceptance criterion
        if delta_E < 0 or random.uniform(0, 1) < math.exp(-beta * delta_E):
            c = c_new
        
        samples.append(c.copy())
    return samples

# Run MCMC
n_steps = 1000
beta = 1.0  # Inverse temperature
samples = metropolis_hastings(c0, H, n_steps, beta)
samples = np.array(samples)

# Histogram of configurations

samples = (1 + samples) // 2  # Convert from -1,1 to 0,1 for easier indexing

M = samples.shape[1]
weights = 2**jnp.arange(M-1, -1, -1)
ss = jnp.dot(samples, weights)

# Count occurrences of each configuration in a histogram
hist, bin_edges = np.histogram(ss, bins=2**M, range=(0, 2**M))

# Plot histogram
plt.bar(bin_edges[:-1], hist, width=1)
plt.xlabel('Configuration index')
plt.ylabel('Counts')
plt.title('Histogram of sampled configurations')
plt.show()

# The energy with the most counts
most_frequent_index = jnp.argmax(hist)
print("Most frequent configuration index:", most_frequent_index)

# Calculate average magnetization
magnetizations = np.mean(samples * 2 - 1, axis=0)  # Convert back to -1,1
avg_magnetization = np.mean(magnetizations)
print("Average magnetization:", avg_magnetization)
