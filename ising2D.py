import netket as nk
import jax.numpy as jnp
from netket.hilbert import Spin
from netket.operator import Ising
from netket.graph import Hypercube
from netket.operator.spin import sigmax, sigmaz
import numpy as np
import jax
import scipy as sp


L = 4
dim = 2
N = L**dim
h = -1.0
J = -1000.0
hz = 0.0

g = nk.graph.Hypercube(length=L,n_dim=dim,pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

ha = nk.operator.Ising(hilbert = hi, graph = g, h=h, J=J)
haz = nk.operator.LocalOperator(hi)
for i in range(N):
    haz += -hz * 0.5 * sigmaz(hi, i)

ham = haz + ha


sol = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=2)
eigvals, eigvecs = sol
E0 = eigvals[0]
En = E0/N
psi0 = eigvecs[:, 0]

# ENERGÍA DEL GAP
E1 = eigvals[1]
gap = E1 - E0
print("Gap energy:", gap)

# MAGNETIZACION ESPERADA POR PARTICULA
# ===============================================

M_z_op = nk.operator.LocalOperator(hi)
for i in range(N):
    M_z_op += 0.5* sigmaz(hi, i)

Mz_sparse_matrix = M_z_op.to_sparse()

Mz = jnp.vdot(psi0, Mz_sparse_matrix @ psi0).real
magnetization_per_site = Mz / N

print("Magnetización esperada: ", Mz)

print("E0/N =", En)
print("Autovectores = ", psi0)
#print("dim espacio Hilbert:", psi0.shape[0])
#print(max(psi0))

