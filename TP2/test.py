from re import X
import numpy as np
from numpy.linalg import qr
from scipy.linalg import solve_triangular

def test():
    A = np.array([[-1, -1],[1, 0],[-1, 1]])
    b = np.array([1, 2, 3])

    Q1,R1 = qr(A,mode='reduced')
    y = Q1.T@b
    x = solve_triangular(R1,y,lower=False)
    return x

print(test())

# def test ():
#     # Banco de pruebas:
#     # A b
#     test_bank = (
#     (np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]]), np.array([2.05, 1.99, 2.02, 1.93])),
#     (np.array([[-1, -1], [1, 0], [-1, 1]]), np.array([4, 2, 3])),
#     (np.array([[-1], [1], [-1]]), np.array([1, 7, 3])),
#     (np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]), np.array([1, 3, -1])),
#     (np.array([[2, 1], [1, 2], [1, 1]]), np.array([7, 5, -1])),
#     (np.array([[1, 1, 1, 1], [1, 5, 6, 4], [1, 4, 8, 6], [1, 2, 1, 3], [1, 3, 2, 8]]), np.array([1, 8, 6, -5, 3])),
#     (np.array([[1, 1, 1, 1], [1, 5, 6, 4], [1, 4, 8, 6], [1, 2, 1, 3], [1, 3, 2, 8]]), np.array([1, 8, 6, -5])),
#     (np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]]), np.array([1, 2, 3, 4])),
#     (np.array([[-1, 1, 0, 3], [-1, 2, -1, -5], [7, -0, -1, 4]]), np.array([4, 2, 0])),
#     (np.array([[-1, 1, 0, 3], [1, 2, -1, -5], [-1, 4, -1, 1]]), np.array([1, 2, 3, 4])),
#     (np.array([[1, 1, 1, 1, 1, 1, -1, -2], [1, 5, 6, 4, 1, 1, -1, 0], [1, 4, 8, 6, 1, 1, 1, 1], [1, 2, 1, 3, 1, 1, 1, 4],
#     [1, 3, 2, 8, 1, 1, 1, -1], [2, 4, 5, 1, 1, -1, 1, 3], [1, 2, 3, 4, 1, 1, 1, 4], [7, 6, 5, 4, 1, -1, 1, 8],
#     [0, -2, 1, 3, 1, -1, 0, 1], [7, 6, 5, 4, 2, -1, 3, 0]]), np.array([1, 8, 6, -5, 1, 2, 3, 4, -6, 1])))

#     print("\n PROGRAMA DE PRUEBA DE leastsq")

#     # Testeo los casos del banco de prueba
#     for i in range(len(test_bank)):
#         print("____________________________________________________")
#         A = test_bank[i][0] # Recupero A para el caso i
#         b = test_bank[i][1] # Recupero b para el caso i

#         print(" " * get_dim(A[0]), "A", " "*(get_dim(A[0]) + 3), "b") # Imprimo A y b
#         for k in range(get_dim(A)):
#             print("[ " + ' '.join(['{:4}'.format(float(item)) for item in A[k]]), "]", end=" ")
#             if k < get_dim(b): # Se considera el caso que b no tenga las
#                 print("\t " * 2, "[", float(b[k]), "]") # dimensiones correctas
#         for j in range(k+1, get_dim(b)):
#             print("\t\t"*(get_dim(A[0])), " [", float(b[j]), "]") # Si b tiene más filas que A, las imprime
#         print("\n")

#         x = leastsq(A, b) # Resuelvo el problema de los cuadrados mínimos con descomposición QR

#         if not isinstance(x, type(-1)): # Verifico que no hubo error
#             y = np.linalg.lstsq(A, b, None) # Resuelvo el problema con numpy
#             print(" x según leastqr" + "\t "*3 + " x según numpy") # Imprimo los resultados
#             for c in range(get_dim(x)):
#                 print("[", '{:{width}.12f}'.format(x[c][0], width=15), "]", "\t"*2, "[", '{:{width}.12f}'.format(y[0][c], width=15), "]")
#         else:
#             print("ERROR: Los parámetros ingresados no cumplen con alguno de los requisitos")
        
#         print("____________________________________________________")
#         print("\nPULSA ENTER PARA CONTINUAR") # Pulsar enter para evaluar el
#         input() # caso que sigue

#     print("FIN DEL PROGRAMA DE PRUEBA")

#     return