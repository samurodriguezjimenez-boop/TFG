import netket as nk
import jax.numpy as jnp
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube
from netket.operator.spin import sigmax, sigmaz
import jax
import scipy as sp
import matplotlib.pyplot as plt

L = 4
dim = 2
N = L**dim

h = jnp.linspace(0, 2, 20)  # transverse field strengths to scan
J = -1.0
gg = h / J
g = gg.transpose()

magnetizations = []
gap = []
for h_val in h:
    g = nk.graph.Hypercube(length=L, n_dim=dim, pbc=False)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    ha = nk.operator.Ising(hilbert=hi, graph=g, h=h_val, J=J)

    sol = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=2)
    eigvals, eigvecs = sol
    E0 = eigvals[0]
    En = E0 / N
    psi0 = eigvecs[:, 0]

    # ENERGÍA DEL GAP
    # ==============================================
    # First excited state energy
    E1 = eigvals[1]
    gapE = E1 - E0
    gap.append(gapE)
    
    # MAGNETIZACION ESPERADA POR PARTICULA
    # ===============================================

    M_z_op = nk.operator.LocalOperator(hi)
    for i in range(N):
        M_z_op += (1/N)*sigmaz(hi, i)

    Mz_sparse_matrix = M_z_op.to_sparse()

    Mz = jnp.vdot(psi0, Mz_sparse_matrix @ psi0).real
    #magnetization_per_site = Mz / N
    magnetizations.append(abs(Mz))

print(max(magnetizations))


plt.plot(-h/J, magnetizations)
plt.xlabel('Coupling ratio g = h/J')
plt.ylabel('Magnetization per site Mz')
plt.title('Magnetization per site vs Transverse Field Strength in 2D Ising Model')
plt.grid()
plt.show()

plt.plot(-h/J, gap)
plt.xlabel('Coupling ratio g = h/J')
plt.ylabel('Energy Gap')
plt.title('Energy Gap vs Transverse Field Strength in 2D Ising Model')
plt.grid()
plt.show()