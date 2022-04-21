import numpy as np
import pandas as pd

from math import sqrt

def thinQRFactorization(A): #en Q1 y R1
    (m,n) = A.shape
    Q1 = np.zeros((m,n))
    R1 = np.zeros((n,n))
    for k in range(n):
        Q1[:,k] = A[:,k]
        for j in range(k):
            Q1[:,k] = Q1[:,k] - (Q1[:,j]@A[:,k])*Q1[:,j]
        Q1[:,k] = Q1[:,k] / sqrt(Q1[:,k].T@Q1[:,k])
        for j in range(k,n):
            R1[k,j] = Q1[:,k].T@A[:,j]
    return Q1,R1

# Q1' = nxm
# b = mx1
def solveTriangular(A,b): # A = R1, b = Q1'*b
    n = A.shape[0]
    x = np.zeros((n,1))
    for k in range(n):
        row = n-1-k 
        x[row,0] = b[row,0]
        for j in range(row+1,n):
            x[row,0] -= A[row,j]*x[j,0]
        x[row,0]/= A[row,row]
    return x

def leastsq(A,b):
    try:
        (m,n) = A.shape
        (o,p) = b.shape
        if m < n:
            print("leastsq: ISSUE: Dimensions of A (mxn)\nm must be greater of equal than n")
            return np.array([[]])
        elif m != o:
            print("leastsq: ISSUE: Dimensions of A (mxn) and b (nx1)\nDimensions of A and b mismatch")
            return np.array([[]])
        elif p != 1:
            print("leastsq: ISSUE: Dimensions of b (nx1)\nb must have 1 column")
            return np.array([[]])
        else:
            Q1,R1 = thinQRFactorization(A)
            x = solveTriangular(R1,Q1.T@b)
            return x
    except:
        print("leastsq: INVALID INPUT")
        return np.array([[]])



A = np.array([[-1,-1],[1,0],[-1,1]])
b = np.array([[1,2,3]]).T

x = leastsq(A,b)
print(x)
print(A@x-b)


# PRODUCTO Y TRANSPUESTA SE PUEDEN USAR FUNCIONES DE NUMPY, QUE LOCO

# TAREAS:
# 1) entender descomp QR -> Usar GramShmid modificado y obtener R de manera inteligente, no calculando QtA
# 2) programar el test (usar librerias)
# 3) emplear el leastqr para q lea y resuelva el sount.txt
# 4) INFORME

# Si es inversible : 
#   Sistema determinado ->  Eliminacion Gaussiana (calc inversa) [ESTA NO]
#                           LU : 
#                           Otras q llevaban al caso triangular:
#                               QR (no es mejor que LU pero permite matrices no cuadradas)
#                               Valores singulares
#                           Cholesky si es definida positiva y simetrica
# Si no es inversible : Valores singulares

# OJO: TENEMOS q implemetar QR, por lo que  


# Funcion sound()
#   toma los datos del archivo sound.txt y los evalua en las funciones con y sin
#   devuelve la matriz A y el vector b
def sonido():
    df  = pd.read_csv('.\TP2\sound.txt',header=None,names=['ti','yi'],dtype={'ti':np.float64,'yi':np.float64},sep=' ')
    ti  = np.array(df['ti'].tolist())
    b  = np.array([df['yi'].tolist()]).T

    A = np.zeros((441000, 6))

    for i in range(len(ti)):
        count = 0 
        if count<6:
            row = []
            row.append(np.cos(1000*np.pi*ti[i]))
            row.append(np.sin(1000*np.pi*ti[i]))
            row.append(np.cos(2000*np.pi*ti[i]))
            row.append(np.sin(2000*np.pi*ti[i]))
            row.append(np.cos(3000*np.pi*ti[i]))
            row.append(np.sin(3000*np.pi*ti[i]))
            A[i] = row
            count+=1

    xsol = leastsq(A, b)
    error = A*xsol - b
    
    return xsol, error

from numpy.linalg import qr
from scipy.linalg import solve_triangular

def test():
    pass
