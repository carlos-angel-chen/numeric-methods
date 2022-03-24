import numpy as np

########################################################################################################################
# binf2dec:
# Convierte un arreglo de un número en punto flotante de 16 bits en un número decimal
# En caso de no recibir un arreglo binario imprime un mensaje de error y no devuelve nada

def binf2dec(arr):
    # Verificación
    if len(arr) != 16:
        print("Por favor ingrese un número binario de 16 dígitos")
        return
    signo = arr[0]                                          # Calculo signo
    if signo != 0 and signo != 1:                           # Reviso si hubo error
        print("Por favor ingrese un número en binario")
        return
    exp = conv_bin_exp(np.array(arr[1:6]))                  # Calculo exponente
    if exp == -1:                                           # Reviso si hubo error
        print("Por favor ingrese un número en binario")
        return
    mantissa = conv_bin_mant(np.array(arr[5 + 1:16]))       # Calculo mantissa
    if mantissa == -1:                                      # Reviso si hubo error
        print("Por favor ingrese un número en binario")
        return

    # Analizo casos
    if exp == 0:
        if mantissa == 0:                                   # Caso 0
            n = 0
        else:                                               # Caso sub-normal
            n = sub_normal(signo, mantissa)
    elif exp == 31:
        if mantissa == 0:                                   # Caso infinito
            n = ((-1)**signo)*np.inf
        else:                                               # Caso not a number
            n = np.nan
    else:
        n = normal(signo, exp, mantissa)                    # Caso normal

    return n

########################################################################################################################


########################################################################################################################
# conv_bin_exp:
# Convierte un arreglo de un número binario en un número decimal entero y positivo
# En caso de no recibir un arreglo en binario devuelve -1

def conv_bin_exp(arr):
    num = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            num = num + 2 ** ((len(arr) - 1) - i)
        elif arr[i] != 0:
            num = -1
            break
    return num

########################################################################################################################


########################################################################################################################
# conv_bin_mant:
# Convierte un arreglo de un número binario en un número racional y positivo
# En caso de no recibir un arreglo en binario devuelve -1

def conv_bin_mant(arr):
    num = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            num = num + 2 ** (-1 - i)
        elif arr[i] != 0:
            num = -1
            break
    return num

########################################################################################################################


########################################################################################################################
# normal:
# Cálculo del número en el caso normal

def normal(s, exp, mantissa):
    exp = exp - SESGO
    n = ((-1) ** s) * (1+mantissa) * (2 ** exp)
    return n

########################################################################################################################


########################################################################################################################
# sub_normal:
# Cálculo del número en caso sub-normal

def sub_normal(s, mantissa):
    exp = 1 - SESGO
    n = ((-1) ** s) * mantissa * (2 ** exp)
    return n

########################################################################################################################


######################################################################################################################################################
# test:
# Función de Prueba. Compara el número convertido con el número verdadero
# En caso de que sean distintos imprime un mensaje de error

def test():

    test_arr = [["Cero: |0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|",                 np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0       ],
                ["Más infinito: |0|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|",         np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.inf  ],
                ["Menos infinito: |1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|",       np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), -np.inf ],
                ["Non Binary: |0|0|0|0|0|0|0|0|0|0|0|0|0|2|0|0|",           np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]), None    ],
                ["Bad size: |0|0|",                                         np.array([0, 0]),                                           None    ],
                ["Not even a number: |h|o|l|a|c|o|m|o|e|s|t|a|s|v|o|s|",    np.array(["holacomoestasvos"]),                             None    ],
                ["Normal (40): |0|1|0|1|0|0|0|1|0|0|0|0|0|0|0|0|",          np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 40      ],
                ["Normal (-12.84): |1|1|0|0|1|0|1|0|0|1|0|0|0|0|0|0|",      np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]), -12.5   ],
                ["Normal (0.09375): |0|0|1|0|1|1|1|0|0|0|0|0|0|0|0|0|",     np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.09375 ],
                ["Sub-Normal (4e-5): |0|0|0|0|0|0|1|0|0|1|0|0|1|0|0|0|",    np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1]), 4e-05   ],
                ["NaN: |1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|",                  np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), np.nan  ]]

    # Casos generales
    for i in range(len(test_arr) - 2):
        print("\n" + test_arr[i][0])
        n = binf2dec(test_arr[i][1])
        if(n != test_arr[i][2]):
            print("ERROR")

    # Caso Sub-Normal
    print("\n" + test_arr[len(test_arr) - 2][0])
    n = binf2dec(test_arr[len(test_arr) - 2][1])
    if not (n >= (test_arr[len(test_arr) - 2][2] - 2**-10)) and ((n <= test_arr[len(test_arr) - 2][2]) + 2**-10):
        print("ERROR")

    # Caso Not a Number
    print("\n" + test_arr[len(test_arr) - 1][0])
    n = binf2dec(test_arr[len(test_arr) - 1][1])
    if not np.isnan(n):
        print("ERROR")

    print("\n" + "Fin del programa de prueba")
    return

######################################################################################################################################################


SESGO = 2 ** (5 - 1) - 1        #Defino el sesgo como una constante para mayor prolijidad
