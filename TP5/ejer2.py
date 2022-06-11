from time import time_ns
import pandas as pd
import numpy as np
from ejer1 import minimi

a = 1
b = 1
c = 1
T1 = 1
T2 = 1

ti = []
yi = []
N = 0

# Coleccion de puntos iniciales a b c T1 T2
X0 = np.array([ [0,0,0,0,0],
                [1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [0,0,0,0,1]])

parameter = np.array([a,b,c,T1,T2])

def read_temp_file():
    df = pd.read_csv('temp.txt',header=None,names=['ti','yi'],sep=' ')
    ti = np.array(df['ti'].tolist())
    yi = np.array(df['yi'].tolist())
    N = len(ti)
    return ti, yi, N

def eje2_func(parameter):
    global ti, yi, N
    result = 0 
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    T1 = parameter[3]
    T2 = parameter[4]
    for i in range(N):
        aux = 1/N * abs(yi[i]-(a + b*np.cos(2*np.pi*ti[i]/T1) + c*np.cos(2*np.pi*ti[i]/T2)))**2
        result = aux + result

    return result

def temperatura():
    global ti, yi, N
    ti, yi, N = read_temp_file()
    param, param_eval = minimi(eje2_func, )