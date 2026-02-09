import os
import numpy as np
import matplotlib.pyplot as plt
#import ising1Dq

os.system('cls')

N = int(input('Número de partículas: '))       #Number of particles in the PBC chain
J = 1       #Ferromagnetic order term modulator
h = 1       #Paramagnetic order term modulator
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
print(H)

eigval, eigvec = np.linalg.eig(H)

v1 = eigvec[:,0]
v2 = eigvec[:,1]
v3 = eigvec[:,2]
v4 = eigvec[:,3]

k1 = False
k2 = False

if k1 == True:

    print('Los autovalores son: ')

    for i in range(2**N):   
        print((eigval[i],0))

if k2 == True:
    print(' y los autovectores: ')
    for i in range(2**N):
        print(eigvec[:,i])

energias = np.sort(eigval)
egap = abs(energias[1] - energias[0])
# print('El gap de energía es: ', egap)

#print(eigvec)

def energy(c, H):
    return np.real(c.conj().T @ H @ c)

conf = []

M = 4
for i in range(M):
    c0 = np.random.choice([0,1],size=2**N) #Estado inicial aleatorio
    print('Estado inicial: ', c0)
    c0 = c0/np.linalg.norm(c0) #Normalizamos el estado inicial
    print('La energía del estado ', i, ' es: ', energy(c0, H))
    conf.append(c0)

print(conf)

print('La energía del estado inicial es: ', energy(c0, H))
