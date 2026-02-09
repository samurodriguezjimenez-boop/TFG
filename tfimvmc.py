# Let's diagonalize the TFIM Hamiltonian with a variational Monte Carlo (VMC) approach
import netket as nk
import jax.numpy as jnp
import jax
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
from numpy import linspace
from flax import nnx
import matplotlib.pyplot as plt

    
N = 20
hi = nk.hilbert.Spin(s=1 / 2, N=N)
h = linspace(0.1, 3.0, 10)
J = -1.0

g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)


energy = []
magnetization = []

for h_val in h:
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h_val, J=J)
    class MF(nnx.Module):
        def __init__(self,*,rngs: nnx.Rngs):
            key = rngs.params()
            self.log_phi_local = nnx.Param(jax.random.normal(key, (1,)))
        def __call__(self, x: jax.Array):
            p = nnx.log_sigmoid(self.log_phi_local * x)
            return 0.5 * jnp.sum(p, axis=-1)
    
    mf_model = MF(rngs=nnx.Rngs(0))
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, mf_model, n_samples=512)
    print(vstate.parameters)

    vstate.init_parameters()
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate)
    gs.run(n_iter=300)

    mf_energy = vstate.expect(H)


    # Magnetization operator
    M = nk.operator.LocalOperator(hi)
    for i in range(N):
        M += (1/N) * sigmaz(hi, i)


    mag = vstate.expect(M)
    energy.append(mf_energy)
    magnetization.append(mag)

print(energy)
print(magnetization)

plotE = False
plotM = True

print("Mean-field VMC energy:", mf_energy)
if plotE == True:
    plt.plot(h, energy, 'o-')
    plt.xlabel('Transverse field h')
    plt.ylabel('Mean-field VMC energy')
    plt.title('Mean-field VMC Energy vs Transverse Field in 1D TFIM')
    plt.grid()
    plt.show()

if plotM == True:
    plt.plot(h, magnetization, 'o-')
    plt.xlabel('Transverse field h')
    plt.ylabel('Mean-field VMC magnetization')
    plt.title('Mean-field VMC Magnetization vs Transverse Field in 1D TFIM')
    plt.grid()
    plt.show()


#print("Mean-field VMC magnetization:", mag)
