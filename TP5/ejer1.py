from scipy.optimize import minimize
import numpy as np

def minimi(func,Xo,tol,itmax):
    return Xo[0,:],func(Xo[0,:]),itmax

# FACU ACA TA LA COSA
func = lambda x: (x[0]-3)**2 + (x[1]-2)**2 + 1
Xo = np.array([[0,0],[0,1],[1,0]])
tol = 1e-6
itmax = 1000

m = minimize(func, Xo[0],method='Nelder-Mead',tol=tol,options={'maxiter':itmax,'initial_simplex':Xo})
x = m.x
f = m.fun

print(m)



def minimi_(func,Xo,tol,itmax):
    # Xo = [x1 x2 x3 ... xn+1].T con xi vectores fila
    # xi = Xo[i,:] = Xo[i]
    # Filas = numero de puntos
    # Columnas = dimension del problema
    n,d = Xo.shape
    if n!=d+1:
        print("ERROR: n must be equal to d+1")
        return Xo[0],func(Xo[0])
    #f = np.zeros((n,1))
    f = np.array([func(xi) for xi in Xo])

    pass

