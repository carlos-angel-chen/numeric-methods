import math
import numpy as np
import matplotlib.pyplot as plt

#Funcion:   solvercrit
#           Aproxima valor de rcrit mediante algoritmo de Newton-Raphson
#           El punto inicial se determina mediante una aproximacion cuadratica a la funcion
#Recibe:    valor de z.
#Devuelve:  r critico
def solvercrit(z):

    #constantes
    Dt = 8e-6
    v = 0.005
    rc = 0.0005
    tm = 5e-5
    k0 = 5.75e-5
    c0 = 5
    r0 = 0.01

    #constantes
    alpha = (4*Dt*c0) / (r0 * (rc + tm)**2)
    beta = (4*Dt) / (v*(rc**2))
    gamma = (2 * Dt) / (rc * k0)
    
    #punto inicial
    y = np.sqrt((alpha+z*beta+gamma-1)/(-1+gamma+beta*z))       

    eps = 10e-9  #cota 

    #calculo de y0
    yexp2 = y**2
    fx = yexp2 * np.log(yexp2) - yexp2 + 1 - alpha + beta * (yexp2 - 1) * z + gamma * (yexp2 - 1)
    dfx = 2*y*(beta*z + gamma + np.log(yexp2))

    y1 = y - fx/dfx     #primera iteracion
    n = 1      

    while(np.abs(y1 - y) > eps or fx > eps):
        n += 1      
        y = y1
        yexp2 = y**2
        fx = yexp2 * np.log(yexp2) - yexp2 + 1 - alpha + beta * (yexp2 - 1) * z + gamma * (yexp2 - 1)
        dfx = 2*y*(beta*z + gamma + np.log(yexp2))
        y1 = y - fx/dfx

    rcrit = y1 * (rc + tm)
    return rcrit    


#Funcion:   rcritplot
#           Grafica r critico para valores de z entre 0 y 0.1
def rcritplot():
    z = np.linspace(0,0.1,1001)
    rcrit = np.empty(len(z))
    for i in range(len(z)):
       rcrit[i] = solvercrit(z[i])

    plt.title('rcrit(z)')
    plt.xlabel('z')
    plt.ylabel('rcrit')
    plt.grid(color='b', ls = '-.', lw = 0.25)
    plt.plot(z, rcrit)

    plt.show()