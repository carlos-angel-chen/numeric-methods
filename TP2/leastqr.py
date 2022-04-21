import numpy as np
import pandas as pd

def leastsq(A,b):
    print("HOLA")

def gramSchmidt(A):
    print("ADIOS")

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(a*b)

def thinQRFactorization(A): #en Q1 y R1
    print("QR FACTORIZATION")

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
def sound():
    df  = pd.read_csv('.\TP2\sound.txt',header=None,names=['ti','yi'],dtype={'ti':np.float64,'yi':np.float64},sep=' ')
    ti  = np.array(df['ti'].tolist())
    b  = np.array(df['yi'].tolist())

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
    
    return A, b


from numpy.linalg import qr
from scipy.linalg import solve_triangular

def test():
    pass
