import numpy as np
import matplotlib.pyplot as plt

def higginsselkov():
    x0 = np.array(2,3)      #s0,p0

    figure, axis = plt.subplots(2, 1)

    for i in (0.48, 0.6, 0.01):
        vinit(i)
        result = ruku4(dx, 0, 600, 0.1, x0)      
        t = result[0]
        x = result[1].T
        axis[0,0].plot(t, x[0])
        axis[1,0].plot(t, x[1])

    plt.title('Higgin-Selkov model')
    axis[0,0].xlabel('t')
    axis[1,0].xlabel('t')
    axis[0,0].ylabel('s')
    axis[1,0].ylabel('p')
    axis[0,0].legend()
    axis[1,0].legend()
    #plt.grid(color='b', ls = '-.', lw = 0.25)  
    plt.show()


def dx(x,t):
    v0 = vinit()
    return np.array(v0 - 0,23 * x[0] * (x[1]**2), 0,23 * x[0] * (x[1]**2) - 0,40 * x[1])

def vinit(v0):
    vinit.v = v0
    return vinit.v

def ruku4(f, t0, tf, h, x0):
    pass