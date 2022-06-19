import numpy as np
import random

def get_nearby_point(xo):
    x = np.array(xo,dtype=np.longdouble)
    for i in range(len(x)):
        x[i] = x[i] * (1 + (random.random()-0.5)/10)
    return x

def evaluate_and_order(X,func):
    f = np.array([func(xi) for xi in X])
    i = f.argsort()
    return X[i],f[i]

def replace_point(X,f,func,R,pos,p,S):
    I = np.array([i for i in range(pos)] + [p] + [i for i in range(pos,p)],dtype=np.int64)
    X[p] = R; f[p] = func(R)
    X, f = X[I], f[I]
    return X, f, S - X[p] + R

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
#Devuelve:  t:     arreglo con los instantes de tiempo
#           x:     aproximaciones numéricas a x (una fila por instante de tiempo)
def minimi(func,grad,xo,tol,itmax):
    random.seed() # Para conseguir n+1 puntos iniciales, se inicializa la librería random

    n = len(xo) # Dimensión del problema
    o = 0       # Índice del punto óptimo
    b = n - 1   # Índice del punto bueno
    p = n       # Índice del punto peor

    X = np.array([get_nearby_point(xo) for i in range(n+1)])
    X,f = evaluate_and_order(X,func)
    S = X[0:n].sum(axis=0)

    for k in range(itmax):
        if np.linalg.norm(X[o]-X[p]) < tol and np.linalg.norm(f[o]-f[p]) < tol:
            break
        M = S / n

        # REFLEXIÓN
        R = 2*M - X[p]; fR = func(R)
        if fR < f[b]:
            if fR > f[o]:
                i = get_pos(fR,f)
                X,f,S = replace_point(X,f,func,R,i,p,S)
            else:
                # EXPANSION
                E = 3*M - 2*X[p]; fE = func(E)
                if fE < f[o]:
                    X,f,S = replace_point(X,f,func,E,o,p,S)
                else:
                    X,f,S = replace_point(X,f,func,R,o,p,S)
        else:
            if fR < f[p]:
                i = get_pos(fR,f)
                X,f,S = replace_point(X,f,func,R,i,p,S)
            else:
                # CONTRACCIÓN
                C1, C2 = (R+M)/2, (X[p]+M)/2
                fC1,fC2 = func(C1),func(C2)
                if fC1<fC2:
                    C,fC = C1,fC1
                else:
                    C,fC = C2,fC2
                if fC < f[p]:
                    i = get_pos(fC,f)
                    X,f,S = replace_point(X,f,func,C,i,p,S)
                else:
                    # ENCOGIMIENTO
                    for i in range(1,n+1):
                        X[i] = (X[i] + X[o])/2
                    X,f = evaluate_and_order(X,func)
                    S = X[0:n].sum(axis=0)
    return X[o],f[o],k
