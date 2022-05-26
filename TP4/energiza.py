import numpy as np
from numpy import linspace
from math import ceil

def ruku4(f,t0,tf,h,x0):
    n = x0.shape[0] # número de componentes de x
    if h <= 0:      # si el paso de integración no es positivo, informa el error
        print("h debe ser mayor a 0")
        return np.zeros(1),np.zeros((n,1))
    N = int(ceil((tf-t0)/h)) # cantidad de pasos. Se toma la función techo para
    # que la h empleada sea a lo sumo tan grande como la pedida
    t = linspace(t0,tf,N+1) # arreglo de tiempos
    x = np.zeros((n,N+1))   # matriz con el resultado
    x[:,0] = x0 # los valores iniciales son dato

    for k in range(N):  # por cada paso del algoritmo...
        tk=t[k]         # se extraen los valores de x(k) y t(k)
        xk=x[:,k]

        # Y se aplica el algoritmo de Runge-Kutta 4
        f1 = f(tk, xk) 
        f2 = f(tk+h/2, xk+f1*h/2)
        f3 = f(tk+h/2, xk+f2*h/2)
        f4 = f(tk+h, xk+f3*h)
        x[:,k+1]=xk+h*(f1+2*f2+2*f3+f4)/6

    x = x.T
    return t,x
