import random


def randmat(p, q, r=1):
    A = [q * [0] for i in range(p)]
    for i in range(p):
        for j in range(q):
            A[i][j] = random.uniform(-r, r)
    return A

def eye(n):
    A = [n * [0] for i in range(n)]
    for i in range(n):
        A[i][i] = 1
    return A

def nb_lig(A): return len(A)

def nb_col(A): return len(A[0])

def multiplier_ligne(A, i, t):
    q = nb_col(A)
    for j in range(q): A[i][j] *= t

def echanger_lignes(A, i, j):
    q = nb_col(A)
    for k in range(q):
        A[i][k], A[j][k] = A[j][k], A[i][k]

def combiner_lignes(A, k, i, t):
    q = nb_col(A)
    for j in range(q): A[k][j] += t * A[i][j]

def chercher_pivot(A, k):
    n = nb_lig(A)
    l = k
    while l < n and A[l][k] == 0: l = l + 1
    if l == n: raise Exception('Pivot non trouvé')
    return l

def pivoter(A, B, k, D):
    n = nb_lig(A)
    l = chercher_pivot(A, k)
    if l != k:
        D = -D
        echanger_lignes(A, k, l)
        echanger_lignes(B, k, l)
    P = A[k][k]
    D = D * P
    multiplier_ligne(B, k, 1 / P)
    multiplier_ligne(A, k, 1 / P)
    for i in range(n):
        if i != k:
            Aik = A[i][k]
            combiner_lignes(A, i, k, -Aik)
            combiner_lignes(B, i, k, -Aik)
    return D

def pivot_gauss(A, B):
    n = nb_lig(A)
    A = [A[i].copy() for i in range(n)]
    B = [B[i].copy() for i in range(n)]
    D = 1
    for i in range(n):
        D = pivoter(A, B, i, D)
    return (A, B, D)

def inverse(A, dbg=False):
    p = nb_lig(A)
    q = nb_col(A)
    if p != q:
        raise Exception('inverser: matrice pas carrée')
    B = eye(p)
    A1, B1, D = pivot_gauss(A, B)
    if dbg:
        print('Déterminant: %.5e' % D)
        return B1

A = randmat(4, 4)
print(A)
print()
B = inverse(A, True)