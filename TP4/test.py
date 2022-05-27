from cProfile import label
import numpy as np
from numpy import linspace
from math import ceil
from scipy.integrate import solve_ivp

from energiza import ruku4
from energiza import higginsselkov

from numpy import pi,cos,exp,sin
import matplotlib.pyplot as plt

v0 = 0.48
def test():
    ######################################################
    # Comparacion entre solve_ivp() y la funcion ruku4() #
    ######################################################

    R = 1e3	            #Valor de la resistencia	
    C = 1e-6	        #Valor de la capacidad
    w = 2.0*pi*1000     #frecuencia angular de la señal de entrada
    A = 1.0		        #amplitud de la señal de entrada
    T = 5*2*pi/w	    #simulo cinco ciclos
    def xsol(t):
        x = -exp(-t/(R*C))+cos(w*t)+w*R*C*sin(w*t)
        x = (A/(1+(w*R*C)**2))*x
        return x
    def dx1(t,x):
        return ((A*cos(w*t)-x)/(R*C))

    x0 = np.zeros(1)

    t4,x4 = ruku4(dx1,0,T,0.0001,x0)
    t = linspace(0,T,int(10e5))
    x = xsol(t)

    sol = solve_ivp(dx1, (0, T), x0, method='RK23', max_step=0.01)

    plt.plot(t,x, label='Explicit')
    plt.plot(t4,x4, label='RK4')
    plt.plot(sol.t, sol.y[0], label='RK23')
    plt.title("Test de la funcion ruku4")
    plt.legend()
    plt.grid(True)
    plt.show()

    ##############################################################################
    # Comparacion entre solve_ivp() y la funcion ruku4(), usando higginsselkov() #
    ##############################################################################
    t0 = 0
    tf = 600                                              
    h = 0.01                                                
    x0 = np.array([2,3])      #s0,p0
    global v0
    v0 = vc = calcVc()

    plt.figure()
    t, result = ruku4(dx, t0, tf, h, x0)   
    result = result.T
    sol1 = solve_ivp(dx, (0, tf), x0, method='RK45', max_step=0.01)

    plt.subplot(2, 1, 1)
    plt.plot(sol1.t, sol1.y[0],'.', label='solve_ivp')
    plt.plot(t, result[0], label='ruku4')
    plt.ylabel('s')
    plt.title("Higgins-Selkov model with v0 = vc")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sol1.t, sol1.y[1],'.', label='solve_ivp')
    plt.plot(t, result[1], label='ruku4')
    plt.xlabel('t')
    plt.ylabel('p')
    plt.legend()
    plt.show()

    plt.figure()
    v0 = 0.48
    t, result = ruku4(dx, t0, tf, h, x0)   
    result = result.T
    sol2 = solve_ivp(dx, (0, tf), x0, method='RK45', max_step=0.01)
 
    plt.subplot(2, 1, 1)
    plt.plot(sol2.t, sol2.y[0],'.', label='solve_ivp')
    plt.plot(t, result[0], label='ruku4')
    plt.ylabel('s')
    plt.title("Higgins-Selkov model with v0 < vc")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sol2.t, sol2.y[1],'.', label='solve_ivp')
    plt.plot(t, result[1], label='ruku4')
    plt.xlabel('t')
    plt.ylabel('p')
    plt.legend()
    plt.show()

    plt.figure()
    v0 = 0.6
    t, result = ruku4(dx, t0, tf, h, x0)   
    result = result.T
    sol3 = solve_ivp(dx, (0, tf), x0, method='RK45', max_step=0.01)
 
    plt.subplot(2, 1, 1)
    plt.plot(sol3.t, sol3.y[0],'.', label='solve_ivp')
    plt.plot(t, result[0], label='ruku4')
    plt.ylabel('s')
    plt.title("Higgins-Selkov model with v0 > vc")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sol3.t, sol3.y[1],'.', label='solve_ivp')
    plt.plot(t, result[1], label='ruku4')
    plt.xlabel('t')
    plt.ylabel('p')
    plt.legend()
    plt.show()


def dx(t,x):
    return np.array([v0 - 0.23 * x[0] * (x[1]**2), 0.23 * x[0] * (x[1]**2) - 0.40 * x[1]])

    #Usar solve_ivp para chequear el correcto funcionamiento de:
    #   ruku4()             OK
    #   higginsselkov()     OK

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

test()