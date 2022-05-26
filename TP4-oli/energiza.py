########################################################################################################################
# Métodos Numéricos(93.54) - 2do Cuatrimestre 2020 - TP N°4 - GRUPO 11
# Ecuaciones diferenciales ordinarias
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci


v0 = 0.54

########################################################################################################################
# kheun:
# Obtiene las soluciones de la EDO ingresada sin calcular derivadas al evaluar f en puntos adecuados
# Recibe los datos incilaes y la funcion y devuelve una matriz con los tiempos tomados y los valores obtenidos a lo largo del programa
# ----------------------------------------------------------------------------------------------------------------------
def kheun(f, x0, t0, tf, h):                                # Recibe la funcion, un array con los valores inciales, el tiempo inicial y final, y el paso
    N = int((tf - t0) / h)                                  # Número de puntos
    t = np.linspace(t0, tf, N + 1)                          # Creo vector con los valores de tiempo
    n = x0.shape[0]                                         # Dimensión del problema
    x = np.zeros((n, N + 1))                                # Creo vector nulo con las dimensiones que debe tener
    x[:, 0] = x0                                            # Le doy los valores inciales

    for k in range(N):
        f1 = h * f(t[k], x[:, k])                           # Calculo de la primer derivada
        f2 = h * f(t[k] + h, x[:, k] + f1)                  # Calculo de la segunda derivada
        x[:, k + 1] = x[:, k] + (f1 + f2) / 2.0             # Se hace promedio de las derivadas

    return t, x                                             # Devuelve la lista con los tiempos tomados y el array con los valores obtenidos
########################################################################################################################


########################################################################################################################
# f:
# Sistema de ecuaciones del ejercicio de higgins-selvok
# ----------------------------------------------------------------------------------------------------------------------
def f(t, x):                                                # Definimos la Funcion que vamos a evaluar con kheun
    return np.array([v0 - 0.23 * x[0] * (x[1] ** 2), 0.23 * x[0] * (x[1] ** 2) - 0.4 * x[1]])
########################################################################################################################


########################################################################################################################
# higginsselkov:
# Resuelve el sistema de ecuaciones pedido en el ejercicio utilizando el método de heun
# ----------------------------------------------------------------------------------------------------------------------
def higginsselkov():
    t0 = 0
    tf = 600                                                # Intervalo al que pertenece t
    h = 0.01                                                # Step
    x0 = np.array([2, 3])                                   # Valores iniciales de s y p respectivamente

    t, v = kheun(f, x0, t0, tf, h)                          # Recibo la lista de tiempos utilizados y la matriz de datos obtenidos utilizando heun

    plotfun(t, v[0, :], "s en función de t", "tiempo", "s")
    plotfun(t, v[1, :], "p en función de t", "tiempo", "p")

    return
########################################################################################################################


########################################################################################################################
# plotfun:
# Grafica la función
# Recibe el dominio, la imagen, el título del gráfico, el nombre de la abscisa y el nombre de la ordenada
# Los últimos tres parámetros son opcionales
# ----------------------------------------------------------------------------------------------------------------------
def plotfun(x, y, title = "y en función de x", xlabel = "x", ylabel = "y"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.show()
    return
########################################################################################################################


########################################################################################################################
# check_perm:
# Comprueba si la solución es de régimen oscilatorio permanente (ROP) en caso de haber mas de una
# solución se comprueba en todas de ellas.
# Recibe v, arreglo con valores de ambas funciones, intervalo t0,tf y step h.
# Devuelve True si se cumple la condición, False en caso contrario
# ----------------------------------------------------------------------------------------------------------------------
def check_perm(v, t0, tf, h):
    i = 0
    for i in range(2):
        # Obtengo mínimos en dos partes diferentes de la función y los comparo
        mthird = min(v[i][int(((tf - t0) / (h * 2))): int((((tf - t0) * (3 / 4)) / h))])  # Tercer cuarto
        mfourth = min(v[i][int((tf - t0) / (h * (4 / 3))): int((tf - t0) / h)])           # Último cuarto
        dif = abs(mthird - mfourth)
        if dif > 0.01:
            print("False")
            return False
        # Repito con máximos
        Mthird = max(v[i][int((tf - t0) / (h * 2)): int((tf - t0) / (h * (4 / 3)))])
        Mfourth = max(v[i][int((tf - t0) / (h * (4 / 3))): int((tf - t0) / h)])
        dif = abs(Mthird - Mfourth)
        if dif > 0.01:
            print("false")
            return False
    print("true")
    return True
########################################################################################################################


########################################################################################################################
# FindVc:
# Resuelve higginsselvok para distintos valores de v0, hasta obtener un valor aproximado de vc
# Devuelve vc
# ----------------------------------------------------------------------------------------------------------------------
def findvc():
    x0 = np.array([2, 3])
    global v0
    v0 = 0.48                       # Defino variables iniciales y el cambio que hara v0 en cada iteración
    vc = v0
    cambio = 0.02
    for i in range(20):
        t, v = kheun(f, x0, 0, 600, 0.01)                                   # Resuelvo kheun para el vo actual
        if check_perm(v, 0, 600, 0.01):                                     # Compruebo si se encuentra en ROP
            print("Sigue siendo oscilación permanente para v0 =", v0, "\n")
            v0 = v0 + cambio                                                # En caso positivo aumento v0
            if v0 == vc:  # Dividimos el cambio en 2 para que no se vuelva al ultimo valor evaluado 1 paso atras
                cambio = cambio / 2
                v0 = v0 - cambio
        else:                                                               # En caso negativo decremento v0
            print("No fue oscilación permanente buscamos un valor mas pequeño. Usando v0 =", v0, "\n")
            vc = v0
            cambio = cambio / 2                                             # También decremento la magnitud del cambio
            v0 = v0 - cambio

        plotfun(t, v[0, :], "s en función de t para el valor de v0 actual", "tiempo", "s")  #Gráfico de s y p
        plotfun(t, v[1, :], "p en función de t para el valor de v0 actual", "tiempo", "p")

    return vc
########################################################################################################################


########################################################################################################################
# testf1:
# Función de prueba 1
# ----------------------------------------------------------------------------------------------------------------------
def testf1a(t, x):
    return np.array([x[0]])
########################################################################################################################


########################################################################################################################
# solf1:
# Función de prueba 1 conseguida analiticamente e^t
# ----------------------------------------------------------------------------------------------------------------------
def solf1(t):
    x = np.zeros(len(t))
    for i in range(len(t)):
        x[i] = np.e**t[i]
    return x
########################################################################################################################
########################################################################################################################
# testf1b
# Sistema de ecuaciones para el ejemplo b
# ----------------------------------------------------------------------------------------------------------------------
def testf1b(t, x):
    return np.array([2*x[0]-3*x[1],-x[0]+4*x[1]])
########################################################################################################################


########################################################################################################################
# test:
# Realiza:
# - EDO simple vs solución analitica
# - Sistema de EDOs vs scipy
# - higginsselkov vs scipy
# ----------------------------------------------------------------------------------------------------------------------
def test():
#test 1a: edo simple y'(t) = y(t)
    t, v = kheun(testf1a, np.array([1]), 0, 9, 0.01)                    # Kheun
    plotfun(t,v[0,:],"y en funcion de t usando kheun","Tiempo", "y(t)")

    vsol = solf1(t)                                                     # Solución exacta y(t)=e^t
    plotfun(t,vsol,"Solucion de y'=y", "tiempo","y(t)")

    err = np.zeros(len(t))                                              # Error
    for j in range(len(t)):
        err[j] = abs(v[0][j] - vsol[j])
    plt.title("error")
    plt.xlabel("tiempo")
    plt.ylabel("y")
    plt.plot(t, err)
    plt.show()

#test 1b: Sistema de EDOs {x'=2x - 3y, y'= -x + 4y}
    t, v = kheun(testf1b, np.array([1,2]), 0, 10, 0.01)                 # Kheun
    plotfun(t,v[0,:],"x(t) usando kheun", "tiempo","x(t)")
    plotfun(t,v[1,:],"y(t) usando kheun", "tiempo","y(t)")

    sol = sci.solve_ivp(testf1b, (0, 10), [1, 2], "RK23")               # Resolución de scipy
    plotfun(sol.t, sol.y[0, :], "x(t) usando scipy", "tiempo", "x(t)")
    plotfun(sol.t, sol.y[1, :], "y(t) usando scipy", "tiempo", "y(t)")

#test 2: higginsselkov vs scipy
    t0 = 0.0
    tf = 600.0
    higginsselkov()                                                     # Higginsselkov
    sol = sci.solve_ivp(f, (t0, tf), [2, 3], "RK23")                    # resolucion de scipy
    plotfun(sol.t, sol.y[0, :], "s usando scipy", "tiempo", "s")
    plotfun(sol.t, sol.y[1, :], "p usando scipy", "tiempo", "p")

    print("fin de prueba")
    return
########################################################################################################################


########################################################################################################################




