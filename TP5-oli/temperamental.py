########################################################################################################################
# Métodos Numéricos(93.54) - 2do Cuatrimestre 2020 - TP N°5 - GRUPO 11
# Optimización
########################################################################################################################

import numpy as np
v = np.loadtxt("temp.txt", delimiter=" ", usecols=(1, 2))
y = v[:, 1]
t = v[:, 0]

########################################################################################################################
# mininmi: Implementa el algoritmo de cuasi-Newton BFGS
# Recibe: - f: Handle de la función a minimizar
#         - vf: Handle del gradiente de la función
#         - x0: Valor cercano al que minimiza la función
#         - tol: Tolerancia
#         - max_i: Número máximo de iteraciones
# ----------------------------------------------------------------------------------------------------------------------
def minimi(f, vf, x0, tol, max_i):
    x_1 = x0
    B = np.diag(np.full(vf(x0).shape, 0.1))             # Creo el B base
    i = 0
    fin = True

    while i < max_i and fin:
        x_0 = x_1                                       # Actualizo x
        d = -np.dot(B, vf(x_0))                         # Calculo d
        if np.linalg.norm(d) >= np.finfo(float).eps:
            g = lambda alpha: f(x_0 + alpha * d)
            a = argmin(g, 0, 1)                         # Calculo alfa que minimice f
            x_1 = x_0 + a * d                           # Calculo nueva x

            if np.linalg.norm(x_1 - x_0) >= tol and np.linalg.norm(d) >= np.finfo(float).eps:        # Me fijo si terminó
                s = a * d
                y = vf(x_1) - vf(x_0)
                B += ((s.T@y + (y.T@B)@y) * (s@s.T)) / (s.T@y) ** 2 - ((B@y)@s.T + s@(y.T@B)) / (s.T@y)  # Aplico BFGS
                i += 1
                # print("iteración ", i, ": min =", x_1)
            else:
                fin = False                             # Si terminó, salgo del loop
        else:
            fin = False

    print("Se realizaron", i, "iteraciones")

    return x_1
########################################################################################################################


########################################################################################################################
# argmin: Realiza la interpolación cuadrática de una función en R y encuentra el mínimo del polinomio
# Recibe: - f: Handle de la función a minimizar
#         - x: Punto de inicio de la minimización (Primer nodo)
#         - h: Estimación de la distancia entre los nodos
# Devuelve: Valor que minimiza el polinomio interpolador
# ----------------------------------------------------------------------------------------------------------------------
def argmin(f, x, h0):
    falta = True
    h = h0
    while falta and h > 1e-10:                      # Se define h = 1e-10 como el menor h soportado
        if f(x) > f(x + h) > f(x + 2*h):            # Si no encuentra un mínimo,
            h = 2*h                                 # Duplica el paso
        elif f(x) < f(x + h) < f(x + 2*h):          # Si se pasa del mínimo,
            h = h / 2                               # Toma la mitad del paso
        else:
            falta = False                           # Encontró un mínimo

    ya = f(x)
    yb = f(x + h)
    yc = f(x + 2*h)
    hmin = (4*yb - 3*ya - yc) / (4*yb - 2*ya - 2*yc) * h        # Calcula el mínimo del polinomio interpolador
    return x + hmin
########################################################################################################################


########################################################################################################################
# temperatura: Busca los valores que minimizan la función fun
# Los handle de fun y su gradiente deben estar definidos en el mismo archivo que esta función
# Devuelve: - Arreglo con los parámetros que minimizan temperatura
#           - Error de ajuste
# ----------------------------------------------------------------------------------------------------------------------
def temperatura():
    print("\nTemperatura")
    va = minimi(fun, vfun, np.array([36.19, 0.0001, -0.0001, 0.8, 1.01]), np.finfo(float).eps, 100)
    err = fun(va)

    return va, err
########################################################################################################################


########################################################################################################################
# Handler de la función a minimizar en temperatura
# ----------------------------------------------------------------------------------------------------------------------
def fun(x):
    return np.sum(funaux(y, t, x[0], x[1], x[2], x[3], x[4]) ** 2)
# ----------------------------------------------------------------------------------------------------------------------
def funaux(yi, ti, a, c, d, T2, T3):
    return yi - (a + c * np.cos(2 * np.pi * ti / T2) + d * np.cos(2 * np.pi * ti / T3))
########################################################################################################################


########################################################################################################################
# Calcula el gradiente de fun a partir de sus derivadas parciales
# ----------------------------------------------------------------------------------------------------------------------
def vfun(x):
    grad = np.zeros(5)
    grad[0] = np.sum(fun_a(y, t, x[0], x[1], x[2], x[3], x[4]))
    grad[1] = np.sum(fun_c(y, t, x[0], x[1], x[2], x[3], x[4]))
    grad[2] = np.sum(fun_d(y, t, x[0], x[1], x[2], x[3], x[4]))
    grad[3] = np.sum(fun_T2(y, t, x[0], x[1], x[2], x[3], x[4]))
    grad[4] = np.sum(fun_T3(y, t, x[0], x[1], x[2], x[3], x[4]))
    return grad
########################################################################################################################
# Derivadas parciales de fun
# ----------------------------------------------------------------------------------------------------------------------
def fun_a(yi, ti, a, c, d, T2, T3):
    return -2 * funaux(yi, ti, a, c, d, T2, T3)
# ----------------------------------------------------------------------------------------------------------------------
def fun_c(yi, ti, a, c, d, T2, T3):
    return -2 * np.cos(2 * np.pi * ti/T2) * funaux(yi, ti, a, c, d, T2, T3)
# ----------------------------------------------------------------------------------------------------------------------
def fun_d(yi, ti, a, c, d, T2, T3):
    return -2 * np.cos(2 * np.pi * ti/T3) * funaux(yi, ti, a, c, d, T2, T3)
# ----------------------------------------------------------------------------------------------------------------------
def fun_T2(yi, ti, a, c, d, T2, T3):
    return -4 * np.pi * c * ti * np.sin(2 * np.pi * ti/T2) * funaux(yi, ti, a, c, d, T2, T3) / T2**2
# ----------------------------------------------------------------------------------------------------------------------
def fun_T3(yi, ti, a, c, d, T2, T3):
    return -4 * np.pi * d * ti * np.sin(2 * np.pi * ti/T3) * funaux(yi, ti, a, c, d, T2, T3) / T3**2
########################################################################################################################


########################################################################################################################
def test():
    tb = np.array([
        np.array(["Esfera corrida en R3", f1, vf1, np.array([0, 0, 0]), np.finfo(float).eps, 100,
                                                                                        np.array([0.0, 1.0, 2.0])   ]),
        np.array(["Función de Beale", fbeale, vfbeale, np.array([0, 0]), np.finfo(float).eps, 1000,
                                                                                        np.array([3.0, 0.5])        ]),
        np.array(["Esfera en R^8", fesfr8, vfesfr8, np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.finfo(float).eps, 100,
                                                                  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]),
        np.array(["Función de cabina", fcab, vfcab, np.array([0, 0]), np.finfo(float).eps, 100,
                                                                                        np.array([1.0, 3.0])        ]),
        np.array(["Función de Matyas", fmat, vfmat, np.array([1, 2]), np.finfo(float).eps, 200,
                                                                                        np.array([0.0, 0.0])        ])])

    for i in range(len(tb)):
        print("\n" + tb[i][0])                                          # Print nombre de la función
        m = minimi(tb[i][1], tb[i][2], tb[i][3], tb[i][4], tb[i][5])    # minimi de la función i de tb
        print(m)
        err = False
        for j in range(len(tb[i][6])):                                  # Compruebo que cada punto minimo encontrado sea
            if np.abs(m[j] - tb[i][6][j]) > 1e-9:                       # aproximadamente el conocido
                print("ERROR: Los valores obtenidos no coinciden con los conocidos")
                print("Mínimo conocido:", tb[i][6])
                err = True
                break
        if not err:
            print("Caso exitoso: El mínimo obtenido coincide con el conocido")
    print(temperatura())                                                 # Función temperatura ejercicio 2
    return
########################################################################################################################


########################################################################################################################
# Función Esfera corrida
# ----------------------------------------------------------------------------------------------------------------------
def f1(x):
    return x[0]**2 + (x[1]-1)**2 + (x[2]-2)**2
########################################################################################################################
# Gradiente de la función Esfera corrida
# ----------------------------------------------------------------------------------------------------------------------
def vf1(x):
   return np.array([2*x[0], 2*(x[1]-1), 2*(x[2]-2)])
########################################################################################################################


########################################################################################################################
# Función Esfera en R^8 utilizada para testear el algoritmo
# ----------------------------------------------------------------------------------------------------------------------
def fesfr8(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2
########################################################################################################################
# Gradiente de la función Esfera
# ----------------------------------------------------------------------------------------------------------------------
def vfesfr8(x):
   return np.array([2*x[0], 2*x[1], 2*x[2], 2*x[3], 2*x[4], 2*x[5], 2*x[6], 2*x[7]])
########################################################################################################################


########################################################################################################################
# Función Beale utilizada para testear el algoritmo
# ----------------------------------------------------------------------------------------------------------------------
def fbeale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
########################################################################################################################
# Gradiente de la función Beale
# ----------------------------------------------------------------------------------------------------------------------
def vfbeale(x):
   return np.array([0.25*(x[1]-1)*((8*x[1]**5 + 8*x[1]**4 + 16*x[1]**3 - 8*x[1] - 24)*x[0] + 21*x[1]**2 + 39*x[1] + 51),
                    0.25*x[0]*(24*x[0]*x[1]**5 + 16*x[0]*x[1]**3 + (63 - 24*x[0])*x[1]**2 + (36-8*x[0])*x[1] - 8*x[0] + 12)])
########################################################################################################################


########################################################################################################################
# Función de cabina utilizada para testear el algoritmo
# ----------------------------------------------------------------------------------------------------------------------
def fcab(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
########################################################################################################################
# Gradiente de la función de cabina
# ----------------------------------------------------------------------------------------------------------------------
def vfcab(x):
   return np.array([10*x[0] + 8*x[1] - 34, 10*x[1] + 8*x[0] - 38])
########################################################################################################################


########################################################################################################################
# Función Matyas utilizada para testear el algoritmo
# ----------------------------------------------------------------------------------------------------------------------
def fmat(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]
########################################################################################################################
# Gradiente de la función de cabina
# ----------------------------------------------------------------------------------------------------------------------
def vfmat(x):
   return np.array([(13*x[0] - 12*x[1])/25, (13*x[1] - 12*x[0])/25])
########################################################################################################################


# print(temperatura())


# print(minimi(f1, vf1, np.array([0, 0, 0]), np.finfo(float).eps, 100))
# print(minimi(fbeale, vfbeale, np.array([0, 0]), np.finfo(float).eps, 1000))
# print(minimi(fesfr8, vfesfr8, np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.finfo(float).eps, 100))
# print(minimi(fcab, vfcab, np.array([0, 0]), np.finfo(float).eps, 100))

test()
