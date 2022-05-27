from pickle import FALSE
import numpy as np
from numpy import linspace
from math import ceil
import matplotlib.pyplot as plt

v0 = 0.48

def higginsselkov():
    t0 = 0
    tf = 600                                               
    h = 0.01                                                
    x0 = np.array([2,3])      #s0,p0

    v0 = calcVc()
    
    t, result = ruku4(dx, t0, tf, h, x0)   

    result = result.T

    figure, axis = plt.subplots(2, 1)
    axis[0,0].plot(t, result[0])
    axis[1,0].plot(t, result[1])
    plt.title('Higgin-Selkov model')
    axis[0,0].xlabel('t')
    axis[1,0].xlabel('t')
    axis[0,0].ylabel('s')
    axis[1,0].ylabel('p')
    axis[0,0].legend()
    axis[1,0].legend()
    #plt.grid(color='b', ls = '-.', lw = 0.25)  
    plt.show()

def dx(t,x):
    return np.array([v0 - 0,23 * x[0] * (x[1]**2), 0,23 * x[0] * (x[1]**2) - 0,40 * x[1]])

def isOsc(t0, tf, h, f):
    dt = (tf-t0)/h
    eps = 0.1
    for i in range(2):
        min1 = min(f[i][int(dt/2):int(dt*3/4)])
        min2 = min(f[i][int(dt*3/4):int(dt)])
        max1 = max(f[i][int(dt/2):int(dt*3/4)])
        max2 = max(f[i][int(dt*3/4):int(dt)])
        if abs(min2 - min1) > eps or abs(max2 - max1) > eps:
            return False
    return True 


def calcVc():
    t0 = 0
    tf = 600                                               
    h = 0.01     
    x0 = np.array([2, 3])
    v0 = vc = 0.48                     

    delta = 0.1
    for i in range(10):
        t, v = ruku4(dx, t0, tf, h, x0)  
        v = v.T                            
        if isOsc(v, 0, 600, 0.01):                                   
            v0 += delta                                                
            if v0 == vc:  
                delta /= 2                                             
                v0 -= delta 
        else:                                                              
            vc = v0
            delta /= 2                                            
            v0 -= delta 
    print(vc)
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