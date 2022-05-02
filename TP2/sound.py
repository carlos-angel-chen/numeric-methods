import numpy as np
import pandas as pd
import leastqr as lqr

from numpy.linalg import qr
from scipy.linalg import solve_triangular


def sonido():
    df  = pd.read_csv('.\TP2\sound.txt',header=None,names=['ti','yi'],dtype={'ti':np.float64,'yi':np.float64},sep=' ')
    ti  = np.array(df['ti'].tolist())
    b  = np.array([df['yi'].tolist()]).T

    A = np.zeros((441000, 6))

    for i in range(len(ti)):
        count = 0 
        if count<6:
            row = []
            row.append(np.cos(1000*np.pi*ti[i]))
            row.append(np.sin(1000*np.pi*ti[i]))
            row.append(np.cos(2000*np.pi*ti[i]))
            row.append(np.sin(2000*np.pi*ti[i]))
            row.append(np.cos(3000*np.pi*ti[i]))
            row.append(np.sin(3000*np.pi*ti[i]))
            A[i] = row
            count+=1

    xsol = lqr.leastsq(A, b)
    error = np.matmul(A,xsol) - b

    # Q1,R1 = qr(A,mode='reduced')
    # y = Q1.T@b
    # xsol_lib = solve_triangular(R1,y,lower=False)
    # error_lib = np.matmul(A,xsol_lib) - b

    return xsol, error

x, x_error = sonido()



