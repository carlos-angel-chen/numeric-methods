import numpy as np
from numpy import linspace
from math import ceil
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

v0 = 0.48

def higginsselkov():
    t0 = 0
    tf = 600                                              
    h = 0.01                                                
    x0 = np.array([2,3])      #s0,p0
    global v0
    v0 = vc = calcVc()

    t, result = ruku4(dx, t0, tf, h, x0)   
    result = result.T
    plot(t, result[0], result[1], "Higgins-Selkov model with v0 = vc")

    result = solve_ivp(dx, (0, tf), x0, method='RK45', max_step=0.01)
    plot(result.t, result.y[0], result.y[1], "scipy vc")

    v0 = 0.48
    t, result = ruku4(dx, t0, tf, h, x0)   
    result = result.T
    plot(t, result[0], result[1], "Higgins-Selkov model with v0 < vc")

    result = solve_ivp(dx, (0, tf), x0, method='RK45', max_step=0.01)
    plot(result.t, result.y[0], result.y[1], "scipy v0 < vc")

    v0 = 0.6
    t, result = ruku4(dx, t0, tf, h, x0)   
    result = result.T
    plot(t, result[0], result[1], "Higgins-Selkov model with v0 > vc")

    result = solve_ivp(dx, (0, tf), x0, method='RK45', max_step=0.01)
    plot(result.t, result.y[0], result.y[1], "scipy v0 > vc")


def dx(t,x):
    return np.array([v0 - 0.23 * x[0] * (x[1]**2), 0.23 * x[0] * (x[1]**2) - 0.40 * x[1]])

def plot(t, s, p, title):
    plt.subplot(2, 1, 1)
    plt.plot(t, s)
    plt.ylabel('s')
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(t, p)
    plt.xlabel('t')
    plt.ylabel('p')

    plt.show()

def isOsc(t0, tf, h, arr):
    dt = (tf-t0)/h
    eps = 0.01
    for i in range(2):
        min1 = min(arr[i][int(dt*2/3):int(dt*5/6)])
        min2 = min(arr[i][int(dt*5/6):int(dt)])
        max1 = max(arr[i][int(dt*2/3):int(dt*5/6)])
        max2 = max(arr[i][int(dt*5/6):int(dt)])
        if abs(min2 - min1) > eps or abs(max2 - max1) > eps:
            return False
    return True 


def calcVc():
    t0 = 0
    tf = 600                                               
    h = 0.01     
    x0 = np.array([2, 3])
    global v0
    v0 = vc = 0.48                     

    delta = 0.01
    for i in range(10):
        t, res = ruku4(dx, t0, tf, h, x0)  
        res = res.T                          
        if isOsc(t0, tf, h, res):                              
            v0 += delta                                                
            if v0 == vc:  
                delta /= 2                                             
                v0 -= delta 
        else:                                                                
            vc = v0
            delta /= 2                                            
            v0 -= delta 
    return vc



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


higginsselkov()