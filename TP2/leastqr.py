import numpy as np

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

from numpy.linalg import qr
from scipy.linalg import solve_triangular

def test():
    pass
