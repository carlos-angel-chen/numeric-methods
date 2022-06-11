from scipy.optimize import minimize
import numpy as np

def minimi(func,Xo,tol,itmax):
    return Xo[:,0]

# FACU ACA TA LA COSA
m = minimize(lambda x: (x[0]-3)**2 + (x[1]-2)**2 + 1, np.array([0,0]),method='Nelder-Mead',tol=1e-6,options={'maxiter':1000})
x = m.x
f = m.fun

print(x,f)



