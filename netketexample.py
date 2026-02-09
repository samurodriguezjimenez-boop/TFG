import netket as nk
import jax.numpy as jnp
import jax
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
from numpy import linspace


N = 20
hi = nk.hilbert.Spin(s=1 / 2, N=N)

Gamma = -1
H = sum([Gamma * sigmax(hi, i) for i in range(N)])
V = -1
H += sum([V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) for i in range(N)])

sp_h = H.to_sparse()
sp_h.shape

from scipy.sparse.linalg import eigsh

eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]

'''

# Now lets run through different transverse fields and get the gap and magnetization
hs = linspace(0.1, 3.0, 10)
gaps = []
magnetizations = []

for h in hs:
    H = sum([-h * sigmax(hi, i) for i in range(N)])
    H += sum([-1 * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) for i in range(N)])
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    eig_vals = jnp.array(eig_vals)
    eig_vecs = jnp.array(eig_vecs)

    # get the ground / first-excited energies and corresponding eigenvectors
    idx = eig_vals.argsort()           # ensure ascending order
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    E_gs = eig_vals[0]
    psi_gs = eig_vecs[:, 0]            # ground-state vector (column)
    E_exc = eig_vals[1]
    psi_exc = eig_vecs[:, 1]           # first excited state

    gap = E_exc - E_gs
    gaps.append(gap)

    M_z_op = nk.operator.LocalOperator(hi)
    for i in range(N):
        M_z_op += (1/N) * sigmaz(hi, i)

    Mz_sparse_matrix = M_z_op.to_sparse()

    Mz = jnp.vdot(psi_gs, Mz_sparse_matrix @ psi_gs).real
    magnetizations.append(abs(Mz))

import matplotlib.pyplot as plt

# And lets print the results
for i in range(len(hs)):
    print(f"h={hs[i]:.3f}, gap={gaps[i]:.6f}, magnetization per site={magnetizations[i]:.6f}")  
    plt.plot(hs, gaps, label='Energy Gap')
    plt.plot(hs, magnetizations, label='Magnetization per site')
    plt.xlabel('Transverse Field Strength h')
    plt.title('1D Transverse Field Ising Model')
    plt.legend()
    plt.grid()
    plt.show()

'''