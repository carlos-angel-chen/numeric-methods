# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:32:30 2021

@author: tiopi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from numba import njit


#%% DATOS DE TP
#Cargamos y mostramos los datos del TP
df  = pd.read_csv('./TP2/sound.txt',header=None,names=['ti','yi'],dtype={'ti':np.float64,'yi':np.float64},sep=' ')
ti  = np.array(df['ti'].tolist())
yi  = np.array(df['yi'].tolist())


plt.close('all')
fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)
ax1.plot(ti,yi,label='med')
ax1.set_xlabel('tiempo [s]')
#ax1.legend()
ax2.plot(ti,yi,label='med')
ax2.set_xlabel('tiempo [s]')
ax2.set_xlim([4.2,4.23])
fig.suptitle('Datos del TP',fontsize=20)
plt.show()
