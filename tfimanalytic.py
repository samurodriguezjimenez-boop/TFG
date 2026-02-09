import os
import numpy as np
import matplotlib.pyplot as plt
#import ising1Dq

os.system('cls')

N = int(input('Número de partículas: '))       #Number of particles in the PBC chain
J = 1       #Ferromagnetic order term modulator
# h = 1000   We now consider h as a variable

h = np.linspace(0,3,301)
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

gap = []


# ========= DEFINICION DEL HAMILTONIANO =========
for hval in h:
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
    Ht = -hval*Ht

    H = Ha + Ht
    #print(H)

    eigval, eigvec = np.linalg.eig(H)

    psi0 = eigvec[:,0]
    energias = np.sort(eigval)
    egap = abs(energias[1] - energias[0])
    gap.append(egap)

k1 = False
k2 = False

if k1 == True:

    print('Los autovalores son: ')

    for i in range(2**N):   
        print(round(eigval[i],0))

if k2 == True:
    print(' y los autovectores: ')
    for i in range(2**N):
        print(eigvec[:,i])


plt.plot(g, gap)
plt.xlabel('Coupling ratio g = h/J')
plt.ylabel('Energy Gap')
plt.title('Energy Gap vs Coupling Ratio in 1D Transverse Field Ising Model')
plt.grid()
plt.show()

#print(eigvec)

