# 1) Usar matrices de nx1 en lugar de arreglos.
# EJ: a = np.zeros((n,1)), en lugar de a = np.zeros(n)
# NOTA: corroborar con a.shape

# 2) DOCUMENTAR!!!

# 3) POLEMICO:
# ¿Podemos usar la función np.linalg.norm() para calcular norma de vectores?
# Sí.

import numpy as np

A = np.zeros((2,3))

row0 = [1,2,3]
row1 = [4,5,6]

A[0] = row0
A[1] = row1

print(A)

A = np.zeros((441000,6))

A[0] = [1,2,3,4,5,6]
A[440998] = [1,2,3,4,5,6]

print(A)
