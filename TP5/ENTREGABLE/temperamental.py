import numpy as np
import random

# Función auxiliar get_nearby_point:
# Dado un punto n-dimensional, de forma aleatoria, consigue otro punto cercano
def get_nearby_point(xo):
    x = np.array(xo,dtype=np.longdouble)
    for i in range(len(x)):
        # Cada coordenada es alterada entre -5% y +5%
        x[i] = x[i] * (1 + (random.random()-0.5)/10)
    return x

# Función auxiliar evaluate_and_order
# Dado un arreglo de puntos y una función, evalúa la función en dichos
# puntos, y los ordena según dichas evaluaciones
def evaluate_and_order(X,func):
    f = np.array([func(xi) for xi in X])
    i = f.argsort()
    return X[i],f[i]

# Función auxiliar replace_point
# Dado un arreglo de puntos, una función, un arreglo de evaluaciones,
# un punto nuevo, y un índice de posición, modifica el arreglo de puntos
# y el de evaluaciones incluyendo el nuevo punto en la posición indicada
def replace_point(X,f,func,R,pos,p,S):
    I = np.array([i for i in range(pos)] + [p] + [i for i in range(pos,p)],dtype=np.int64)
    X[p] = R; f[p] = func(R)
    X, f = X[I], f[I]
    return X, f, S - X[p] + R

# Función auxiliar get_pos
# Dado un arreglo ordenado y un escalar, devuelve el índice donde debería
# colocarse dicho escalar en el arreglo.
def get_pos(fR,f):
    for i,fi in enumerate(f):
        if fR < fi:
            break
    return i

#Función:   minimi
#           Encuentra el mínimo local de una función dada, mediante
#           el método de optimización de Nelder-Mead
#Recibe:    func: Función que se quiere optimizar
#           grad: Gradiente de la función anterior (no se emplea en este método)
#           xo:   Punto inicial para iterar
#           tol:  Tolerancia máxima entre la función evaluada en el punto óptimo y el peor
#           itmax:Numero máximo de iteraciones.
#Devuelve:  Xo: Punto óptimo final
#           fo: Función evaluada en dicho punto
#           k:  Número de iteraciones que llevó realizó
def minimi(func,grad,xo,tol,itmax):
    random.seed() # Para conseguir n+1 puntos iniciales, se inicializa la librería random

    n = len(xo) # Dimensión del problema
    o = 0       # Índice del punto óptimo
    b = n - 1   # Índice del punto bueno
    p = n       # Índice del punto peor

    # Se consiguen los n+1 puntos iniciales
    X = np.array([get_nearby_point(xo) for i in range(n+1)])
    # Se los ordena según el valor de f evaluada en dichos puntos
    X,f = evaluate_and_order(X,func)
    S = X[0:n].sum(axis=0) # Se calcula la suma de los primeros n

    for k in range(itmax): # En cada iteración...
        # ... si la distancia entre el punto óptimo y el peor es menor a la tolerancia,
        # y además la diferencia entre la función evaluada en dichos puntos es también menor
        # a la tolerancia:
        if np.linalg.norm(X[o]-X[p]) < tol and np.linalg.norm(f[o]-f[p]) < tol:
            # Termina el algoritmo
            break
        M = S / n   # Punto medio para calcular reflexiones

        # REFLEXIÓN
        R = 2*M - X[p]; fR = func(R)
        if fR < f[b]: # Reflexión mejor que el bueno
            if fR > f[o]: # Reflexión peor que el óptimo
                # Me quedo con la reflexión
                i = get_pos(fR,f) # Calculo su posición en el arreglo
                X,f,S = replace_point(X,f,func,R,i,p,S) # Lo coloco, eliminando el peor

            else: # Reflexión mejor que el óptimo
                # EXPANSION
                E = 3*M - 2*X[p]; fE = func(E)
                if fE < f[o]: # Expansión mejor que el óptimo
                    # Me quedo con la expansión
                    X,f,S = replace_point(X,f,func,E,o,p,S) # La coloco, eliminando el peor
                else: # Expansión peor que el óptimo
                    # Me quedo con la reflexión
                    X,f,S = replace_point(X,f,func,R,o,p,S) # La coloco, eliminando el peor
        else: # Reflexión peor que el bueno
            if fR < f[p]: # Reflexión mejor que el peor
                # Me quedo con la reflexión
                i = get_pos(fR,f) # Calculo su posición en el arreglo
                X,f,S = replace_point(X,f,func,R,i,p,S) # Lo coloco, eliminando el peor
            else: # Reflexión peor que el peor
                # CONTRACCIÓN
                C1, C2 = (R+M)/2, (X[p]+M)/2
                fC1,fC2 = func(C1),func(C2)
                # Calculo la mejor C
                if fC1<fC2:
                    C,fC = C1,fC1
                else:
                    C,fC = C2,fC2
                if fC < f[p]: # Contracción mejor que el peor
                    # Me quedo con la contracción
                    i = get_pos(fC,f) # Calculo su posición en el arreglo
                    X,f,S = replace_point(X,f,func,C,i,p,S) # Lo coloco, eliminando el peor
                else: # Contracción peor que el peor
                    # ENCOGIMIENTO
                    for i in range(1,n+1): # A todos los puntos que no sean el óptimo
                        X[i] = (X[i] + X[o])/2 # Se reemplazan por uno más cercano al óptimo
                    # Se los ordena según el valor de f evaluada en dichos puntos
                    X,f = evaluate_and_order(X,func)
                    S = X[0:n].sum(axis=0) # Se calcula la suma de los primeros n
    return X[o],f[o],k

import pandas as pd

a = 1
b = 1
c = 1
T1 = 1
T2 = 1

ti = []
yi = []
N = 0

# Coleccion de puntos iniciales a b c T1 T2
X0 = np.array([10,10,10,11,11])

tol = 1e-12
itmax = 1000

parameter = np.array([a,b,c,T1,T2])

def read_temp_file():
    df = pd.read_csv('temp.txt',header=None,names=['ti','yi'],sep=' ')
    ti = np.array(df['ti'].tolist())
    yi = np.array(df['yi'].tolist())
    N = len(ti)
    return ti, yi, N

def eje2_func(parameter):
    global ti, yi, N

    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    T1 = parameter[3]
    T2 = parameter[4]

    aux = abs(yi-(a + b*np.cos(2*np.pi*ti/T1) + c*np.cos(2*np.pi*ti/T2)))**2
    return aux.sum()/N

def temperatura():
    global ti, yi, N
    ti, yi, N = read_temp_file()
    param, param_eval, cant_ite = minimi(eje2_func, None, X0, tol, itmax)
    #print(cant_ite)

    return param, param_eval