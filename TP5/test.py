from ejer1 import *
from ejer2_V2 import *
import numpy as np

def test():
    testbench = np.array([
        np.array(["Funcion Esfera R3", f1, [[10, 11, 12], [-11, -10, -15], [4, -7, 6], [-9, -7, 6]], [0,0,0]], dtype=object),
        np.array(["Funcion polinomio suma de diferente potencias R4", f2, [[-1, -1, -1, -1], [1, -1, -1, -1], [1, 1, -1, -1], [1, 1, 1, -1], [1, 1, 1, 1]], [0,0,0,0]], dtype=object),
        np.array(["Funcion Beale R2", f3, [[3, 3], [-1, -3], [4, -1.5]], [3, 0.5]], dtype=object),
        np.array(["Funcion de Bohachevsky R2", f4, [[3, 3], [-1, -2], [2, -1.5]], [0, 0]], dtype=object),
        np.array(["Funcion Matyas R2", f5, [[3, 3], [-1, -3], [4, -1.5]], [0,0]], dtype=object),
        np.array(["Funcion Zakharov R3", f6, [[-5, -5, -5], [5, 5, 5], [-5, 5, 5], [5, 3, 4]], [0,0,0]], dtype=object)], dtype=object)
    
    tol = 1e-9
    eps = 1e-6
    itmax = 5000

    failed = 0
    passed = 0 

    print("--------------------------------------")
    print("TEST MINIMI()") 
    print("--------------------------------------")

    for i in range(len(testbench)):
        print("--------------------------------------") 
        print(testbench[i][0])     
        print("--------------------------------------")                                      
        x, fx, it = minimi(testbench[i][1], None, testbench[i][2][0], tol, itmax)    # minimi de la función i de testbench
        spx,spfun,spnit = minimi_(testbench[i][1], None, testbench[i][2][0], tol, itmax)    # scipy.optimize.minimize() de la función i de testbench

        print("X0 function = ", testbench[i][3])        
        print("X0 minimi() = ", x)   
        print("X0 scipy.optimize.minimize() = ", spx)   
        print("Iterations minimi(): ", it)           
        print("Iterations scipy.optimize.minimize(): ", spnit)        
        if np.linalg.norm(spx - x) > eps:
            print("FAILED")
            failed += 1
        else:
            print("PASSED")
            passed += 1
            
    print("PASSED: ", passed)
    print("FAILED: ", failed)

    print("--------------------------------------")
    print("TEST TEMPERATURA()") 
    print("--------------------------------------")

    x, fx = temperatura()    
    spx,spfun,spnit = minimi_(eje2_func, None, X0, tol, itmax)    # minimi de la función i de testbench
    print("X0 temperatura() = ", x)   
    print("X0 scipy.optimize.minimize() = ", spx) 
    if(np.linalg.norm(x - spx)) > eps: 
        print("TEMPERATURA(): FAILED")
    else:
        print("TEMPERATURA(): PASSED")
    return

#Test: Funcion Esfera R3
def f1(x):
    return x[0]**2 + x[1]**2 + x[2]**2

#Test: Funcion polinomio suma de diferente potencias R4
def f2(x):
    return x[0]**2 + abs(x[1])**3 + x[2]**4 + abs(x[3])**5 

#Test: Funcion Beale R2
def f3(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*(x[1]**2))**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2

#Test: Funcion de Bohachevsky R2
def f4(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0] + 4*np.pi*x[1]) + 0.3

#Test: Funcion Matyas R2
def f5(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

#Test: Funcion Zakharov R3
def f6(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + (0.5*x[0] + x[1] + 1.5*x[2])**2 + (0.5*x[0] + x[1] + 1.5*x[2])**4

test()