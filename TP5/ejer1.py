from scipy.optimize import minimize
import numpy as np

def minimi(func,grad,Xo,tol,itmax):
    m = minimize(func, Xo[0],method='Nelder-Mead',tol=tol,options={'maxiter':itmax,'initial_simplex':Xo})
    #return Xo[0,:],func(Xo[0,:]),itmax
    return m.x,m.fun,m.nit
# FACU ACA TA LA COSA
# func = lambda x: (x[0]-3)**2 + (x[1]-2)**2 + 1
# Xo = np.array([[0,0],[0,1],[1,0]])
# tol = 1e-6
# itmax = 1000

# m = minimize(func, Xo[0],method='Nelder-Mead',tol=tol,options={'maxiter':itmax,'initial_simplex':Xo})
# x = m.x
# f = m.fun

# print(m)



def minimi_(func,grad,Xo,tol,itmax):
    # Xo = [x1 x2 x3 ... xn+1].T con xi vectores fila
    # xi = Xo[i,:] = Xo[i]
    # Filas = numero de puntos
    # Columnas = dimension del problema
    n,d = Xo.shape
    if n!=d+1:
        print("ERROR: n must be equal to d+1")
        return Xo[0],func(Xo[0])

    f = np.array([func(xi) for xi in Xo])
    i = f.argsort()
    f = f[i]
    Xo = Xo[i]

    x = Xo[0]
    o = 0
    b = n-2
    p = n-1
    S = np.sum(Xo[0:d],axis=0)
    for k in range(itmax):
        if np.linalg.norm(Xo[o]-Xo[p]) < tol and np.linalg.norm(f[o]-f[p]) < tol:
            print(Xo)
            print(f)
            break
        M = S/d

        #REFLECTION
        R = 2*M - Xo[p]
        fR = func(R)
        if fR < f[b]:
            if f[o] < fR:
                l=1
                while fR >= f[l]:
                    l += 1
                I = np.array([i for i in range(l)]+[p]+[i for i in range(l,p)],dtype=np.int64)
                Xo[p] = R
                Xo = Xo[I]#np.array([Xo[0:l], R, Xo[l:(b+1)]])
                f = f[I]#np.array
                S = S - Xo[p] + R

            else:
                #EXPANDO
                E = 3*M-2*Xo[p]
                fE = func(E)
                if fE < f[o]:
                    I = np.array([p]+[i for i in range(p)],dtype=np.int64)
                    Xo[p] = E
                    f[p] = fE
                    Xo = Xo[I]
                    f = f[I]
                    S = S - Xo[p] + E
                else:
                    I = np.array([p]+[i for i in range(p)],dtype=np.int64)
                    Xo[p] = R
                    f[p] = fR
                    Xo = Xo[I]
                    f = f[I]
                    S = S - Xo[p] + R
        else:
            if fR < f[p]:
                Xo[p] = R
                f[p] = fR
                #S = S - Xo[p] + R
            else:
                # CONTRAIGO
                C1 = (Xo[p]+M)/2
                C2 = (R+M)/2
                fC1 = func(C1)
                fC2 = func(C2)
                if fC1 < fC2:
                    C = C1
                    fC = fC1
                else:
                    C = C2
                    fC = fC2
                if fC < f[p]:
                    l = 0
                    while fC >= f[l]:
                        l += 1

                    I = np.array([i for i in range(l)]+[p]+[i for i in range(l,p)],dtype=np.int64)
                    Xo[p] = C
                    f[p] = fC
                    Xo = Xo[I]
                    f = f[I]
                    S = S - Xo[p] + C
                else:
                    # ENCOJO
                    for l in range(1,n):
                        Xo[l] = (Xo[l]+x)/2
                        f[l] = func(Xo[l])
                    i = f.argsort()
                    f = f[i]
                    Xo = Xo[i]
                    S = np.sum(Xo[0:d],axis=0)
        x = Xo[o]   
    return Xo[o],f[o],k

