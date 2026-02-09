import os
import numpy as np
import matplotlib.pyplot as plt
#import ising1Dq

# os.system('cls')

N = int(input('Número de partículas: '))       #Number of particles in the PBC chain
J = 1.0       #Ferromagnetic order term modulator
h = 1.0       #Paramagnetic order term modulator
g = h/J     #Relative order modulator


sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
I = np.identity(2)

def sj(a,j,N): #define matriz de Pauli particula j-esima en direccion a, N particulas
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    I = np.identity(2)
    if a == 1:
        sxj = 1
        for i in range(j-1):
            sxj = np.kron(sxj,I)
        sxj = np.kron(sxj,sx)
        for i in range(N-j):
            sxj = np.kron(sxj,I)
        return sxj
    if a == 2:
        syj = 1
        for i in range(j-1):
            syj = np.kron(syj,I)
        syj = np.kron(syj,sy)
        for i in range(N-j):
            syj = np.kron(syj,I)
        return syj
    if a == 3:
        szj = 1
        for i in range(j-1):
            szj = np.kron(szj,I)
        szj = np.kron(szj,sz)
        for i in range(N-j):
            szj = np.kron(szj,I)
        return szj

def dictoarr(dic): #convierte los valores de diccionarios en arrays
    a = list(dic.values())
    b = np.array(a)
    return b

sxs = {}
szs = {}

# Definimos las matrices de Pauli en cada direccion para cada particula
for i in range(1,N+1,1): # IMPORTANTE: CREAR VARIABLES SIN A PRIORI
    sxi = f"sx{i}"
    szi = f"sz{i}"
    Mx = sj(1,i,N)
    Mz = sj(3,i,N)
    sxs[sxi] = Mx
    szs[szi] = Mz

sxsval = dictoarr(sxs)
szsval = dictoarr(szs)

# sz1 = szs['sz1']

# ========= DEFINICION DEL HAMILTONIANO =========

Ha = 0
for i in range(N):
    if i == N-1:
        Ha = Ha + sxsval[i]@sxsval[0]
    else:
        Ha = Ha + sxsval[i]@sxsval[i+1]
Ha = -J * Ha
#print(sxsval)

Ht = 0
for i in range(N):
    Ht = Ht + szsval[i]
Ht = -h*Ht

H = Ha + Ht
# print(H)

# Random initial configuration: simple system Nh = 2, Nv = N
Nh = 2  # Number of hidden units
Nv = N  # Number of visible units (same as number of spins)

Nv0 = np.random.choice([-1, 1], size=Nv)  # Random binary configuration
Nh0 = np.random.choice([-1, 1], size=Nh)  # Random binary configuration for hidden units

# Randomize the RBM parameters
# W = np.random.normal(0, 0.01, size=(Nv, Nh))  # Weights

W = np.ones((Nv, Nh))  # Small positive weights to encourage learning

print("Initial RBM parameters (weights):", W)
a = np.zeros(Nv)  # Visible biases
b = np.zeros(Nh)   # Hidden biases
print("Initial visible configuration (spin state):", Nv0)
print("Initial hidden configuration:", Nh0)

# Calculate the energy of the initial configuration
def energy(v,h):
    return -np.sum(np.dot(v, a)) -np.sum(np.dot(h, b)) - np.sum(np.dot(np.dot(v, W),h))

# Initial configuration:
print("Initial energy:", energy(Nv0,Nh0))

    
    
E0 = energy(Nv0,Nh0)

'''
for j in range(Nh):
        r = np.random.rand()
        Phj = np.exp(np.dot(v, W)*Nh0[j] + b[j]) / (np.exp(np.dot(v, W)*Nh0[j] + b[j]) + np.exp(-np.dot(v, W)*Nh0[j] - b[j]))  # Hidden unit probabilities
        if r >= Phj:     Nh0[j] = 1
        else:            Nh0[j] = -1
'''