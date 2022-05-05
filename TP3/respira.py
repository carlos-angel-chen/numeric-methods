import math
import numpy as np

def solvercrit(z):
    Dt = 8e-6
    v = 0.005
    rc = 0.0005
    tm = 5e-5
    k0 = 5.75e-5
    c0 = 5
    r0 = 0.01

    alpha = (4*Dt*c0) / (r0 * (rc + tm)**2)
    beta = (4*Dt) / (v*(rc**2))
    gamma = (2 * Dt) / (rc * k0)

    y = 1       #punto inicial

    eps = 10e-6  #cota 

    yexp2 = y**2
    fx = yexp2 * np.log(yexp2) - yexp2 + 1 - alpha + beta * (yexp2 - 1) * z + gamma * (yexp2 - 1)
    dfx = 2*y*(beta*z + gamma + np.log(yexp2))

    print(fx)
    print(dfx)

    y1 = y - fx/dfx
    print(y1)

    print(np.abs(y1 - y))

    n = 0

    while(np.abs(y1 - y) > eps):
        n += 1
        y = y1
        yexp2 = y**2
        fx = yexp2 * np.log(yexp2) - yexp2 + 1 - alpha + beta * (yexp2 - 1) * z + gamma * (yexp2 - 1)
        dfx = 2*y*(beta*z + gamma + np.log(yexp2))
        y1 = y - fx/dfx

    print("n")
    print(n)
    print(y1)
    rcrit = y1 * (rc + tm)
    return rcrit

print(solvercrit(0.1))