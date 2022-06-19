import numpy as np
import random

def minimi_(func,grad,Xo,tol,itmax):
    from scipy.optimize import minimize
    m = minimize(func, Xo,method='Nelder-Mead',tol=tol,options={'maxiter':itmax})
    #return Xo[0,:],func(Xo[0,:]),itmax
    return m.x,m.fun,m.nit

def get_nearby_point(xo):
    x = np.array(xo,dtype=np.double)
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

def minimi(func,grad,xo,tol,itmax):
    # Xo = [x1 x2 x3 ... xn+1].T con xi vectores fila
    # xi = Xo[i,:] = Xo[i]
    # Filas = numero de puntos
    # Columnas = dimension del problema
    random.seed()
    n = len(xo)
    o = 0
    b = n - 1
    p = n

    X = np.array([get_nearby_point(xo) for i in range(n+1)])
    X,f = evaluate_and_order(X,func)
    S = X[0:n].sum(axis=0)

    print("Inicial")
    printXf(X,f)
    for k in range(itmax):
        print(f'iteration number {k+1}')
        if np.linalg.norm(X[o]-X[p]) < tol and np.linalg.norm(f[o]-f[p]) < tol:
            break
        M = S / n
        print(f'M = {M}')

        # REFLEXIÓN
        R = 2*M - X[p]; fR = func(R)
        print(f'R = {R}\tf(R) = {fR}')
        if fR < f[b]:
            print("f(R) < f(B)")
            if fR > f[o]:
                print("f(R) > f(O)")
                i = get_pos(fR,f)
                print(f'ME QUEDO CON R, POSICION {i+1}')
                X,f,S = replace_point(X,f,func,R,i,p,S)
            else:
                print("f(R) <=f(O)")
                E = 3*M - 2*X[p]; fE = func(E)
                print(f'E = {E}\tf(E) = {fE}')
                if fE < f[o]:
                    print("f(E) < f(O)")
                    print(f'ME QUEDO CON E, POSICION {o+1}')
                    X,f,S = replace_point(X,f,func,E,o,p,S)
                else:
                    print("f(E) >=f(O)")
                    print(f'ME QUEDO CON R, POSICION {o+1}')
                    X,f,S = replace_point(X,f,func,R,o,p,S)
        else:
            print("f(R) >=f(B)")
            if fR < f[p]:
                print("f(R) < f(P)")
                i = get_pos(fR,f)
                print(f'ME QUEDO CON R, POSICION {i+1}')
                X,f,S = replace_point(X,f,func,R,i,p,S)
            else:
                # CONTRACCIÓN
                print("f(R) >=f(P)")
                C1, C2 = (R+M)/2, (X[p]+M)/2
                fC1,fC2 = func(C1),func(C2)
                print(f'C1 = {C1}\tf(C1) = {fC1}')
                print(f'C2 = {C2}\tf(C2) = {fC2}')
                if fC1<fC2:
                    C,fC = C1,fC1
                else:
                    C,fC = C2,fC2
                print(f'C = {C}\tf(C2) = {fC}')
                if fC < f[p]:
                    print("f(C) < f(P)")
                    i = get_pos(fC,f)
                    print(f'ME QUEDO CON C, POSICION {i+1}')
                    X,f,S = replace_point(X,f,func,C,i,p,S)
                else:
                    # ENCOGIMIENTO
                    print("f(C) >=f(P)")
                    print("ENCOJO!!!")
                    for i in range(1,n+1):
                        X[i] = (X[i] + X[o])/2
                    X,f = evaluate_and_order(X,func)
                    S = X[0:n].sum(axis=0)
        printXf(X,f)
    return X[o],f[o],k

def printXf(X,f):
    for i in range(len(X)):
        print(f'{i+1}: x = {X[i]} \t f = {f[i]}')

def func(X):
    return X[0]**2 + X[1]**2

xo,fo,k = minimi(func,None,np.array([10,10]),1e-9,1000)
print(xo,fo,k)
