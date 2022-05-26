import numpy as np
from numpy import linspace
from math import ceil

def ruku4(f,t0,tf,h,x0):
    n = x0.shape[0]
    if h <= 0:
        return np.zeros(1),np.zeros((n,1))
    N = int(ceil((tf-t0)/h)) # h final a lo sumo tan grande como la original
    t = linspace(t0,tf,N+1)
    x = np.zeros((n,N+1))
    x[:,0] = x0

    for k in range(N):
        tk=t[k]
        xk=x[:,k]
        f1 = f(tk, xk)
        f2 = f(tk+h/2, xk+f1*h/2)
        f3 = f(tk+h/2, xk+f2*h/2)
        f4 = f(tk+h, xk+f3*h)

        x[:,k+1]=xk+h*(f1+2*f2+2*f3+f4)/6

    x = x.T
    return t,x
