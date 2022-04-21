import numpy as np
import pandas as pd

from math import sqrt

def thinQRFactorization(A): #en Q1 y R1
    print("QR FACTORIZATION")
    (m,n) = A.shape
    Q1 = np.zeros((m,n))
    R1 = np.zeros((n,n))

    for k in range(n):
        R1[k,k] = np.linalg.norm(A[:,k])#sqrt(A[:,k].T@A[:,k])
        Q1[:,k] = A[:,k]/R1[k,k]
        for j in range(k+1,n):
            R1[k,j] = Q1[:,k].T@A[:,j]
            A[:,j] = A[:,j] - Q1[:,k]*R1[k,j]
            #print(Q1[:,k]*R1[k,j])

        #print("LOL")
        #print(A)
        #print("LOL")

    return Q1,R1

def leastsq(A,b):
    q,r = np.linalg.qr(A, mode = 'reduced')
    print(A)
    print("")
    print("")
    print(q)
    print("")
    print(r)
    print("")
    print(q@r)
    print("")
    print("")

    q,r = thinQRFactorization(A)
    print(q)
    print("")
    print(r)
    print("")
    print(q@r)

    print("HOLA")

def gramSchmidt(A):
    print("ADIOS")

A = np.array([[-1,-1,2],[1,0,77],[-1,1,5]])
b = np.array([7,6,4]).T
leastsq(A,b)
# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# print(a*b)




def solveTriangular(A,b): # A = R1, b = Q1'*b
    print("solve triangular")


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
