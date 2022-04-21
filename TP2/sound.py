import numpy as np
import pandas as pd


df  = pd.read_csv('.\TP2\sound.txt',header=None,names=['ti','yi'],dtype={'ti':np.float64,'yi':np.float64},sep=' ')
ti  = np.array(df['ti'].tolist())
b  = np.array(df['yi'].tolist())

A = []