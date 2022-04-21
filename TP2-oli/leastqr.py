########################################################################################################################
# Métodos Numéricos(93.54) - 2do Cuatrimestre 2020 - TP N°2 - GRUPO 11
# Se busca resolver el problema de cuadrados mínimos mediante descomposición QR
########################################################################################################################

import numpy as np

########################################################################################################################
# leastsq:
# Resuelve el problema de cuadrados mínimos usando descomposición QR (argmin||A.x-b||)
# Recibe la matriz A y el vector b. Devuelve el vector x.
# Requisitos: - Que la cantidad de filas de A sea mayor o igual a la de columnas
#             - Que A sea de rango completo
#             - Que las dimensiones de A y b sean compatibles
# En caso de que no se cumplan los requisitos devuelve -1
# ----------------------------------------------------------------------------------------------------------------------
def leastsq(A, b):
    # Verificación
    if not verify(A, b):                                # Verifico que se cumplan los requisitos
        return -1

    # Factorización QR
    QR = desc_qr(A)                                     # Realizo la descomposición QR
    Q1 = QR[0]                                          # Recupero Q1 de m*n y ortonormal de A
    R1 = QR[1]                                          # Recupero R1 triangular superior de n*n

    # Resuelvo para encontrar x mínima                  # R1.x = Q^T.b
    C = np.dot(np.transpose(Q1), b)                     # C = Q1^T.b
    x = rev_subs(R1, C)                                 # R1.x = C

    return x
########################################################################################################################


########################################################################################################################
# desc_qr:
# Realiza la descomposición QR reducida usando Gram-Schmidt. Devuelve Q1 y R1
# Asume que se cumplen los requisitos de leastsq
# ----------------------------------------------------------------------------------------------------------------------
def desc_qr(A):
    m = get_dim(A)                                          # Defino las dimensiones m y n
    n = get_dim(A[0])

    Q1 = np.zeros(shape=(m, n))                             # Creo las matrices base
    R1 = np.zeros(shape=(n, n))
    for i in range(n):
        aux = A[:, i]
        for k in range(i):                 # Calculo las proyecciones (Q1[:, k] nunca va a ser nula porque es ortogonal)
            aux = aux - (np.dot(A[:, i], Q1[:, k])/norma2(Q1[:, k])) * Q1[:, k]
        Q1[:, i] = normalize(aux)                           # Normalizo la columna
        R1[i, i] = norma2(aux)                              # Calculo la diagonal de R1
        j = i + 1
        while(j < n):
            R1[i, j] = np.dot(A[:, j], Q1[:, i])            # Calculo la esquina superior de R1
            j += 1

    return Q1, R1
########################################################################################################################


########################################################################################################################
# rev_subs:
# Resuelve un sistema por sustitución hacia atrás (A.x = y). Devuelve x
# La matriz A ingresada debe ser triangular superior, y debe ser un vector
# ----------------------------------------------------------------------------------------------------------------------
def rev_subs(A, y):
    m = get_dim(A)                                                          # Defino m
    x = np.zeros(shape=(m, 1))                                              # Crea el vector base
    for i in range(m):
        d = 0
        for j in range(i):
            d += A[m - i - 1, m - j - 1] * x[m - j - 1]                     # Calcula los términos conocidos
        x[m - i - 1] = (1/A[m - i - 1, m - i - 1])*(y[m - i - 1] - d)       # Calcula una de las componente

    return x
########################################################################################################################


########################################################################################################################
# verify:
# Revisa que se cumplan los requisitos de leastsq
# Devuelve True si se cumplen, False sino
# ----------------------------------------------------------------------------------------------------------------------
def verify(A, b):
    r = True
    m = get_dim(A)                                                       # Defino las dimensiones m y n
    n = get_dim(A[0])
    if m < n:                                                            # Verifico que m >= n
        print("POR FAVOR INGRESE UNA MATRIZ CON MAYOR O IGUAL CANTIDAD FILAS QUE COLUMNAS")
        r = False
        if np.linalg.matrix_rank(A) != m:                                # Verifico que sea de rango completo (con m)
            print("POR FAVOR INGRESE UNA MATRIZ DE RANGO COMPLETO")
    elif np.linalg.matrix_rank(A) != n:                                  # Verifico que sea de rango completo (con n)
            print("POR FAVOR INGRESE UNA MATRIZ DE RANGO COMPLETO")
            r = False
    if len(b) != m:                                                      # Verifico que la dimensión de b tenga sentido
        print("POR FAVOR CHEQUEE LAS DIMENSIONES DE b CON RESPECTO A LAS DE A")
        r = False

    return r
########################################################################################################################


########################################################################################################################
# get_dim:
# Atrapa la excepción al calcular la dimensión de un arreglo en caso de que sea 1
# Si es otra cosa, devuelve -1
# ______________________________________________________________________________________________________________________
def get_dim(M):
    try:
        n = len(M)                                          # Calculo la dimensión del arreglo
    except TypeError:                                       # Si tira excepción, me fijo si es un número
        if isinstance(M, int) or isinstance(M, float):
            n = 1                                           # Si es, devuelve 1
        else:
            n = -1                                          # Si no, devulve -1 (no va a ocurrir el contexto de este TP)

    return n
########################################################################################################################


########################################################################################################################
# norma2:
# Devuelve la norma 2 del vector ingresado
# ----------------------------------------------------------------------------------------------------------------------
def norma2(v):
    n = 0
    for i in range(len(v)):
        n = n + (v[i])**2               # Elevo cada componente al cuadrado y las sumo
    return np.sqrt(n)                   # Devuelvo la raíz de la suma de los cuadrados
########################################################################################################################


########################################################################################################################
# normalize:
# Devuelve la versión normalizada del vector ingresado
# ----------------------------------------------------------------------------------------------------------------------
def normalize(v):
    v = v/norma2(v)

    return v
########################################################################################################################


########################################################################################################################
# test:
# Prueba el correcto funcionamiento de leastsq, mostrando su solución junto a la de numpy
# Imprime ambos vectores junto con los parámetros A y b ingresados
# El programa le pedirá que toque la tecla enter para analizar el caso siguiente
# Se revisan muchos tamaños distintos y los tres requisitos de leastsq
# ----------------------------------------------------------------------------------------------------------------------
def test():

    # Banco de pruebas
    #                  A                                                                            b
    test_bank = (
    (np.array([[1.02, 1], [1.01, 1], [0.94, 1], [0.99, 1]]),                           np.array([2.05, 1.99, 2.02, 1.93])),
    (np.array([[-1, -1], [1, 0], [-1,  1]]),                                           np.array([4, 2, 3])),
    (np.array([[-1], [1], [-1]]),                                                      np.array([1, 7, 3])),
    (np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]),                                      np.array([1, 3, -1])),
    (np.array([[2, 1], [1, 2], [1, 1]]),                                               np.array([7, 5, -1])),
    (np.array([[1, 1, 1, 1], [1, 5, 6, 4], [1, 4, 8, 6], [1, 2, 1, 3], [1, 3, 2, 8]]), np.array([1, 8, 6, -5, 3])),
    (np.array([[1, 1, 1, 1], [1, 5, 6, 4], [1, 4, 8, 6], [1, 2, 1, 3], [1, 3, 2, 8]]), np.array([1, 8, 6, -5])),
    (np.array([[-1, -1], [-1, -1], [-1,  -1], [-1, -1]]),                              np.array([1, 2, 3, 4])),
    (np.array([[-1, 1, 0, 3], [-1, 2, -1, -5], [7, -0, -1, 4]]),                       np.array([4, 2, 0])),
    (np.array([[-1, 1, 0, 3], [1, 2, -1, -5], [-1, 4, -1, 1]]),                        np.array([1, 2, 3, 4])),
    (np.array([[1, 1, 1, 1, 1, 1, -1, -2], [1, 5, 6, 4, 1, 1, -1, 0], [1, 4, 8, 6, 1, 1, 1, 1], [1, 2, 1, 3, 1, 1, 1, 4],
               [1, 3, 2, 8, 1, 1, 1, -1], [2, 4, 5, 1, 1, -1, 1, 3], [1, 2, 3, 4, 1, 1, 1, 4], [7, 6, 5, 4, 1, -1, 1, 8],
               [0, -2, 1, 3, 1, -1, 0, 1], [7, 6, 5, 4, 2, -1, 3, 0]]), np.array([1, 8, 6, -5, 1, 2, 3, 4, -6, 1])))

    print("\n PROGRAMA DE PRUEBA DE leastsq")

    # Testeo los casos del banco de prueba
    for i in range(len(test_bank)):
        print("__________________________________________________________________")
        A = test_bank[i][0]                                                 # Recupero A para el caso i
        b = test_bank[i][1]                                                 # Recupero b para el caso i

        print("   " * get_dim(A[0]), "A", "   "*(get_dim(A[0]) + 3), "b")            # Imprimo A y b
        for k in range(get_dim(A)):
            print("[ " + ' '.join(['{:4}'.format(float(item)) for item in A[k]]), "]", end=" ")
            if k < get_dim(b):                                                # Se considera el caso que b no tenga las
                print("\t " * 2, "[", float(b[k]), "]")                   # dimensiones correctas
        for j in range(k+1, get_dim(b)):
            print("\t\t"*(get_dim(A[0])), " [", float(b[j]), "]")            # Si b tiene más filas que A, las imprime
        print("\n")

        x = leastsq(A, b)                    # Resuelvo el problema de los cuadrados mínimos usando descomposición QR

        if not isinstance(x, type(-1)):                                 # Verifico que no hubo error

            y = np.linalg.lstsq(A, b, None)                             # Resuelvo el problema con numpy
            print("  x según leastqr" + "\t "*3 + "   x según numpy")     # Imprimo los resultados
            for c in range(get_dim(x)):
                print("[", '{:{width}.12f}'.format(x[c][0], width=15), "]", "\t"*2, "[", '{:{width}.12f}'.format(y[0][c], width=15), "]")

        else:
            print("ERROR: Los parámetros ingresados no cumplen con alguno de los requisitos")

        print("__________________________________________________________________")
        print("\nPULSA ENTER PARA CONTINUAR")                           # Pulsar enter para evaluar el caso que sigue
        input()

    print("FIN DEL PROGRAMA DE PRUEBA")

    return
########################################################################################################################
