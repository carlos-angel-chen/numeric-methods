import numpy as np
from math import sqrt

import warnings
warnings.filterwarnings("error")


# Función thinQRFactorization
# Recibe: matriz no singular de dimensiones mxn, con m>=n
# Devuelve: matrices Q1 y R1, correspondientes a la factorización QR reducida de la matriz de entrada
def thinQRFactorization(A):
    (m,n) = A.shape      # Se obtienen las dimensiones de la matriz de entrada...
    Q1 = np.zeros((m,n)) # ...para definir las matrices Q1 y R1, inicialmente en 0
    R1 = np.zeros((n,n))

    # Metodo Gram-Schmidt (tradicional)
    for k in range(n):      # Por cada columna de Q1...
        Q1[:,k] = A[:,k]    # se extrae una nueva columna de A
        for j in range(k):  # y por cada una de las columnas anteriores de Q1 (ahora versores) ...
            Q1[:,k] = Q1[:,k] - (Q1[:,j]@A[:,k])*Q1[:,j] # ... se le resta la proyección de la nueva columna en cada versor
        try:
            Q1[:,k] = Q1[:,k] / sqrt(Q1[:,k].T@Q1[:,k]) # Se normaliza la nueva columna
        except:
             print("leastsq: INVALID INPUT. Rank(A) < dim(A) \nb A must not be singular")
             return np.array([[]]), np.array([[]])
        for j in range(k,n): # Por cada una de las siguientes columnas
            R1[k,j] = Q1[:,k].T@A[:,j] # Se calcula el valore de R1
    return Q1,R1

# Función solveTriangular
# Recibe: matriz de coeficientes A cuadrada (nxn), y vector de terminos independientes (nx1) 
# A debe ser triangular superior
# Devuelve: matriz (nx1) solución al sistema Ax=b
def solveTriangular(A,b):
    n = A.shape[0]      # Se obtiene el tamaño de A...
    x = np.zeros((n,1)) # ...para crear la matriz resultado, inicialmente en 0
    for k in range(n):  # Por cada fila del resultado
        row = n-1-k     # (comenzando por la última)
        x[row,0] = b[row,0] # Se iguala al término independiente
        for j in range(row+1,n): # Y por cada una de las filas ya calculadas
            x[row,0] -= A[row,j]*x[j,0] # Se sustrae su contribución con la incógnita actual
        x[row,0]/= A[row,row] # Finalmente, se divide por el coeficiente de la incógnita actual
    return x

# Función leastsq
# Recibe: matrices de numpy A (tamaño mxn) y b (tamaño mx1) correspondientes
# al sistema Ax = b
# Devuelve: matriz x de tamaño nx1 que minimice la norma del error
# e = Ax-b
def leastsq(A,b):
    try:
        (m,n) = A.shape
        (o,p) = b.shape
        if m < n:       # Comuncar error... Si A tiene mas columnas que filas;
            print("leastsq: ISSUE: Dimensions of A (mxn)\nm must be greater of equal than n")
            return np.array([[]])
        elif m != o:    #... Si A y b tienen un número distinto de filas
            print("leastsq: ISSUE: Dimensions of A (mxn) and b (nx1)\nDimensions of A and b mismatch")
            return np.array([[]])
        elif p != 1:    #... o si b tiene mas (o menos) de 1 columna
            print("leastsq: ISSUE: Dimensions of b (nx1)\nb must have 1 column")
            return np.array([[]])
        else: # Si las entradas son válidas
            Q1,R1 = thinQRFactorization(A) # Se factoriza a A en Q1 y R1 (factorización QR reducida)
            if Q1.shape == (0,0) and R1.shape == (0,0):
                print("leastsq: ISSUE: rank(A) < dim(A) \nb A must not be singular")
                x = np.array([[]])
                return x
            x = solveTriangular(R1,Q1.T@b) # y se resuelve el sistema de ecuaciones R1x = Q1'b
            xSum = np.sum(x)
            if np.isnan(xSum):
                print("leastsq: ISSUE: rank(A) < dim(A) \nb A must not be singular")
                x = np.array([[]])

            return x

    except: # Mensaje enviado en caso de que las entradas no sean válidas 
        print("leastsq: INVALID INPUT")
        return np.array([[]])

import pandas as pd
# Funcion sounido()
#   Toma los datos del archivo sound.txt y los evalua en las funciones coseno y seno,
#   obteniendo asi la matriz A y b
#   Hace llamado a la funcion leastsq(A,b) y les envia la matriz A y b
#   Devuelve la xsol y el error
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




# Funcion test()
#   Testbench para funcion leastsq()
#   Analiza 3 casos sin errores y 5 casos que no cumplen condiciones.
#   Condiciones para factorizacion QR:
#       A de rango completo
#       m >= n
#       dim(b) = m

def test():
    # Test bank: A, b
    testCases = (
    #4x2
    (np.array([[5.1, 0], [3.4, 1], [-4, 1], [0.11, 1]]), np.array([[5, 1.8, 9.9, -1.5]])), 
    #3x1 
    (np.array([[-1], [0], [-1]]), np.array([[1, 5, 2]])),          
    #3x3                           
    (np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]), np.array([[0.7, 3, 4.1]])),        
    #4x2        rank(A)<n         
    (np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), np.array([[5, 1.8, 9.9, -1.5]])),     
    #2x3        m<n     
    (np.array([[4, 1, 1], [1, 0, 1]]), np.array([[1, 2, -4]])),                   
     #2x3        m<n     rank(A)<n            
    (np.array([[1, 1, 1], [1, 1, 1]]), np.array([[1, 2, -4]])),     
    #4x2        Dimensions of A (mxn) and b (nx1) mismatch                         
    (np.array([[5.1, 0], [3.4, 1], [-4, 1], [0.11, 1]]), np.array([[5, 1.8, 9.9]])),  
    #4x2        b with more columns       
    (np.array([[5.1, 0], [3.4, 1], [-4, 1], [0.11, 1]]), np.array([[5, 1.8, 9.9, -1.5], [1, 0, 0, 1]])),        
    )

    eps = 10**(-6)
    passed = failed = 0

    print("___________________________________________")
    print("TEST BENCH")

    for i in range(len(testCases)):
        A = testCases[i][0]
        b = testCases[i][1]
        
        #b = testCases[n] if n < n_casos_particulares else createAb()
        print("TEST N°: ", i)
        print("A =")
        print(A)
        print("b =")
        print(b)
        
        x = leastsq(A, b.T)

        if x.size == 0:
            (m,n) = A.shape
            (o,p) = b.shape
            if m < n or np.linalg.matrix_rank(A) < n or m != o or p != 1:
                print("ISSUE DETECTED. PASSED \n")
                passed +=1
            else:
                print("ISSUE NOT DETECTED. FAILED \n")
                failed += 1
        else:
            print("x =")
            print(x)
            x_prime = np.linalg.lstsq(A,b.T, rcond=None)
            print("x_prime =")
            print(x_prime[0])
            diff = np.linalg.norm(x - x_prime[0])

            if diff < eps:
                print("norm(x-x_prime) < eps. PASSED \n")
                passed +=1
            else:
                print("norm(x-x_prime) >= eps. FAILED \n")
                failed +=1

    print("CASES: ", len(testCases))
    print("PASS: ", passed)
    print("FAIL: ", failed)

print(test())

