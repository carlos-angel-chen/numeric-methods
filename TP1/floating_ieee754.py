from math import copysign,isnan
from re import X
from sympy import false

TBITS = 16                  # Número de bits totales que contiene el número flotante a desarrollar
EBITS = 5                   # Número de bits que componen el exponente
MBITS = TBITS - (1+EBITS)   # Número de bits que componen la mantisa

BIAS = 2**(EBITS-1)-1       # Sesgo del exponente

BINARY16_BITS_NAN = [True]*TBITS                        # Representación de "NaN" en punto flotante
BINARY16_BITS_INF = [False]+[True]*EBITS+[False]*MBITS  # Representación de "-inf" en punto flotante
BINARY16_BITS_MINF= [True]*(1+EBITS)+[False]*MBITS      # Representación de "+inf" en punto flotante
BINARY16_BITS_0   = [False]*TBITS                       # Representación de +0.0 en punto flotante
BINARY16_BITS_M0  = [True] + [False]*(EBITS + MBITS)    # Representación de -0.0 en punto flotante

# custom_log2
# Recibe: número en formato punto flotante
# Devuelve: función piso del logaritmo en base 2 de dicho número
def custom_log2(float_num: float): 
    if isnan(float_num):            # log2(NaN) = NaN
        log =  float('nan')
    elif float_num < 0:             # log2(x) = NaN si x<0
        log =  float('nan')
    elif float_num == float('inf'): # log2(inf) = inf
        log =  float('inf')
    elif float_num == 0:            # log2(0) = -inf
        log =  float('-inf')
    elif float_num == 1:            # log2(1) = 0
        log =  0
    elif float_num < 1:             # Caso 0 < x < 1
        log = 0
        while float_num < 1:            # El número será multiplicado por 2 tantas veces como
            log -= 1                    # sea necesario hasta que tenga la forma 1.mantisa
            float_num *= 2              # Esa cantidad de veces representa el exponente negativo
    else:                           # Caso 1 < x < inf
        log = 0
        while float_num >= 2:           # El número será dividido por 2 tantas veces como
            log += 1                    # sea necesario hasta que tenga la forma 1.mantisa
            float_num /= 2              # esa cantidad de veces representa el exponente positivo
    return log

# float_to_list
# Recibe: número en formato punto flotante
# Devuelve: lista de bits que representan el número en formato float de 16 bits
def float_to_list(float_num: float):
    if isnan(float_num):              # Se analizan por separado los casos: "NaN"
        return BINARY16_BITS_NAN
    elif float_num == float('inf'):   # +inf
        return BINARY16_BITS_INF
    elif float_num == float('-inf'):  # -inf
        return BINARY16_BITS_MINF

    sign = copysign(1,float_num) # Obtenemos el signo del númeor ingresado. El motivo
                                 # por el cual se emplea esta función es porque las
                                 # comparaciones como <, <=, > y >= no pueden distinguir
                                 # entre +0.0 y -0.0
    if float_num == 0:  # Se analiza por separado los casos:
        if sign == 1:                   # +0.0
            return BINARY16_BITS_0
        else:                           # -0.0
            return BINARY16_BITS_M0

    exp = custom_log2(abs(float_num)) # Obtenemos el log2 del número ingresado

    if exp > 2**EBITS-1 - BIAS:       # Si el exponente es superior al máximo representable
        if sign == 1:                 # el número representado será infinito, con el signo correspondiente
            return BINARY16_BITS_INF
        else:
             return BINARY16_BITS_MINF

    man_list = []                
    float_num = abs(float_num)      # Reemplazamos al número ingresado por su valor absoluto

    if exp < 1 - BIAS:              # Si el exponente es menor al mínimo representable, se trata de un
                                    # número "sub normal"
        float_num /= 2**(1 - BIAS)  # Preparamos al número para ser representado mediante el menor
        exp = 0                     # exponente permitido (si bien el exponente indicará "0" en binario)

    else:                           # Si el número es "normal"
        float_num /= 2**exp         # Lo dividimos para que tenga la forma 1.mantisa
        float_num -= 1              # Le sustraemos el 1, el cual será implícito
        exp += BIAS                 # Le agregamos el sesgo al exponente para representarlo

    for pos in range(MBITS):        # Por cada uno de los bits que compondrán la mantisa
        float_num*=2                # En binario, movemos a la izquierda una posición todos los bits
        if float_num >= 1:          # Si a la izquierda del punto decimal hay un 1
            float_num-=1                # lo restamos 
            man_list += [True]          # y agregamos un 1 lógico a la mantisa
        else:                       # Si a la izquierda del punto decimal hay un 0
            man_list += [False]         # agregamos un 0 lógico a la mantisa

    exp_list = []
    for pos in range(EBITS):            # Por cada uno de los bits que compondrán el exponente
        if exp >= 2**(EBITS-(1+pos)):   # Si el bit analizado es 1 (si el número es mayor o igual a "100...00b", con cantidad de 0 decrecientes)
            exp-= 2**(EBITS-(1+pos))        # lo restamos
            exp_list+=[True]                # y agregamos un 1 lógico al exponente
        else:                           # Si el bit analizado es 0
            exp_list+=[False]               # agregamos un 0 lógico al exponente

    return [sign == -1] + exp_list + man_list   # Armamos la lista juntando todas sus partes

# list_to_float
# Recibe: una lista de bits que representan un número de punto flotante de 16 bits
# Devuelve: el mismo número, en formato float nativo
def list_to_float(list_num):
    if list_num == BINARY16_BITS_INF:    # Analizamos por separado los casos: +inf
        return float('inf')
    elif list_num == BINARY16_BITS_MINF: # -inf
        return float('-inf')

    sign = list_num[0]                      # Separamos las partes del número: signo
    exp_list = list_num[1:(EBITS+1)]        # exponente
    man_list = list_num[(EBITS+1):TBITS]    # mantisa

    if exp_list == [True]*EBITS and man_list != [False]*MBITS: # Si el exponente tiene solo 1s, y la mantisa no es sólamente 0s
        return float('nan')                                    # se trata de un "NaN"

    man = 0
    for position in range(MBITS):                    # Por cada bit de la mantisa (del menos significativo al mas significativo)
        man += (2**position)*man_list[-(1+position)] # sumamos su aporte al número
    man /= 2**(MBITS)                                # y corremos todos los bits detrás del punto decimal, obteniendo 0.mantisa

    if exp_list == [False]*EBITS:   # De tratarse de un número "sub normal"
        exp = 1                         # el exponente considerado será 1
    else:                           # De tratarse de un número "normal"
        man += 1                        # Agregamos el 1 implícito de la mantisa
        exp = 0
        for position in range(EBITS):   # y armamos el exponente sumando la contribución de cada bit
            exp += (2**position)*exp_list[-(1+position)]
    exp -= BIAS                     # Luego, le restamos el sesgo al exponente (obteniendo el valor real del mismo)

    return ((-1)**sign * man * 2**exp)  # Fórmula para calcular el valor de un número de punto flotante "a mano"

    

class binary16:

    def __init__(self,float_num: float):
        if type(float_num) is not float:    # Si el número que le pasaron al constructor no es float
            float_num = float(float_num)    # lo castea a float para asegurarse que el programa funcione
                                            # de manera uniforme
        self.bits = float_to_list(float_num)    # Transformamos la variable de entrada en un arreglo de bools,
                                                # correspondientes al valor en punto flotante de 16 bits
        self.d = list_to_float(self.bits)       # Finalmente, transformamos este nuevo valor en punto flotante
                                                # nativo de python (64 bits), perdiendose la precisión del valor original
    
    def __mul__(self,other):
        return binary16(self.d * other.d)   # No se requiere consideraciones especiales a la hora de realizar el producto
                                            # de dos números en formato punto flotante nativo (python respeta las reglas
                                            # pedidas, por ejemplo: inf*0=NaN )

    def __truediv__(self,other):
        if other.d == 0:                    # Sí se debe tener en cuenta el caso del 0 como divisor (ya que python produce un error)

            if self.d == 0:                 # Si AMBOS operandos son 0:   0/0=NaN
                div = float('nan')
            elif isnan(self.d):             # Si el dividendo es NaN:   NaN/0=NaN
                div = float('nan')
            else:                           # Si el dividendo es distinto de 0 o NaN (esto incluye valores "normales", "sub normales"
                                            # +inf y -inf):             var/0=inf, con el signo correspondiente
                if copysign(1,self.d) == copysign(1,other.d): # operandos con mismo signo
                    div = float('inf')              # resultado positivo (y viceversa)
                else:
                    div = float('-inf')
        else:
            div = self.d / other.d          # Si el divisor NO es 0, entonces python realiza la división de forma acorde a lo espeado
                                            # (por ejemplo: inf/inf=NaN )
        return binary16(div)

    # NOTA:
    # Podrían haberse definido de manera análoga los operadores __imul__ (*=) y __itruediv__ (/=)
    # Sin embargo, en caso de no realizarse una sobrecarga para dichos operadores, por defecto
    # python emplea los métodos __mul__ y __truediv__ y luego realiza la asignación, por lo cual
    # dicha sobrecarga de operadores es innecesaria

    def __eq__(self,other):
        if isnan(self.d) and isnan(other.d): # Debe analizarse por separado el caso de que ambas variables sean "NaN", debido a que
            return True                      # python por defecto considera que toda variables es distinta a un "NaN", incluyendo otro "NaN"
        elif (self.d == 0.0) and (other.d == -0.0):
            if(self.bits[0] == other.bits[0]):
                return True
            else:
                return False
        return (self.d == other.d)  # Si solo una de las variables es NaN, la respuesta será False, tal como se espera.
                                    # Si ninguna es NaN, la comparación funciona de la forma esperada

    def __ne__(self,other):
        if isnan(self.d) and isnan(other.d): # Análogamente al caso anterior, comparaciones de desigualdad entre un "NaN"
            return False                     # y otra variable (incluyendo "NaN") devolverán siempre "True". Por ello, se analiza este caso
        elif (self.d == 0.0) and (other.d == -0.0):
            if(self.bits[0] == other.bits[0]):
                return False
            else:
                return True
        return (self.d != other.d)



#########################
#       TEST BENCH      #
#########################

#############################################################################################################
#   Se utilizo la siguiente pagina web para verificar los resultados obtenidos de los siguientes arrays     #
#       https://oletus.github.io/float16-simulator.js/                                                      #
#       mul_verification[]                                                                                  #
#       div_verification[]                                                                                  #
#       equ_verification[]                                                                                  #
#       no_equ_verification[]                                                                               #
#############################################################################################################


def test():
    #Todos los numeros para usar en el testeo
    num_test = [
                ["float", "sub-float", "0", "-0", "inf", "-inf", "nan"],
                [5.984634, 0.345788, 0.0, -0.0, float('inf'), float('-inf'), float('nan')]
                ]

    mul_verification = [35.8125, 2.068359375, 0.0, -0.0, float('inf'), float('-inf'), float('nan'), 
                            0.1195068359375, 0.0, -0.0, float('inf'), float('-inf'), float('nan'),
                                0.0, -0.0, float('nan'), float('nan'), float('nan'), 
                                    0.0, float('nan'), float('nan'), float('nan'),
                                        float('inf'), float('-inf'), float('nan'),
                                            float('inf'), float('nan'), 
                                                float('nan')]

    div_verification = [1.0,17.296875,float('inf'),float('-inf'),0.0,-0.0,float('nan'),
                        0.0577392578125,1.0,float('inf'),float('-inf'),0.0,-0.0,float('nan'),
                        0.0,0.0,float('nan'),float('nan'),0.0,-0.0,float('nan'),
                        -0.0,-0.0,float('nan'),float('nan'),-0.0,0.0,float('nan'),
                        float('inf'),float('inf'),float('inf'),float('-inf'),float('nan'),float('nan'),float('nan'),
                        float('-inf'),float('-inf'),float('-inf'),float('inf'),float('nan'),float('nan'),float('nan'),
                        float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')]

    equ_verification = [True,False,False,False,False,False,False,
                            True,False,False,False,False,False,
                                True,False,False,False,False,
                                    True,False,False,False,
                                        True,False,False,
                                            True,False,
                                                True]

    no_equ_verification = [False,True,True,True,True,True,True,
                                False,True,True,True,True,True,
                                    False,True,True,True,True,
                                        False,True,True,True,
                                            False,True,True,
                                                False,True,
                                                    False]

    #Tomo los numeros de test y lo paso a 16 bits
    num_test_bin16 = []
    for i in range(len(num_test[1])):
        num_test_bin16.append(binary16(num_test[1][i]))
    
    #Creo arreglo de los resultados multiplicando todos los casos
    mul_test = []
    for i in range(len(num_test_bin16)):
        for j in range(i, len(num_test_bin16)):
            mul_test.append(num_test_bin16[i]*num_test_bin16[j])
    
    #Creo arreglo de los resultados comparando con == todos los casos
    equ_test = []
    for i in range(len(num_test_bin16)):
        for j in range(i, len(num_test_bin16)):
            equ_test.append(num_test_bin16[i]==num_test_bin16[j])
    
    #Creo arreglo de los resultados comparando con != todos los casos
    no_equ_test = []
    for i in range(len(num_test_bin16)):
        for j in range(i, len(num_test_bin16)):
            no_equ_test.append(num_test_bin16[i]!=num_test_bin16[j])
    
    #Creo arreglo de los resultados multiplicando todos los casos
    div_test = []
    for i in range(len(num_test_bin16)):
        for j in range(len(num_test_bin16)):
            div_test.append(num_test_bin16[i]/num_test_bin16[j])

    mul_equ_noequ_cases = 28*3
    div_cases = 49
    total_PASS = mul_equ_noequ_cases + div_cases

    pass_cases = 0 
    fail_cases = 0
    for i in range(len(mul_test)):
        if mul_test[i].d == mul_verification[i]:
            pass_cases+=1
        elif isnan(mul_test[i].d) and isnan(mul_verification[i]):
            pass_cases+=1
        else:
            fail_cases+=1

    for i in range(len(equ_test)):
        if equ_test[i] == equ_verification[i]:
            pass_cases+=1
        else:
            fail_cases+=1

    for i in range(len(no_equ_test)):
        if no_equ_test[i] == no_equ_verification[i]:
            pass_cases+=1
        else:
            fail_cases+=1

    for i in range(len(div_test)):
        if div_test[i].d == div_verification[i]:
            pass_cases+=1
        elif isnan(div_test[i].d) and isnan(div_verification[i]):
            pass_cases+=1
        else:
            fail_cases+=1

    if pass_cases == total_PASS:
        print("You pass all cases")
    else:
        print("You fail in {fail_cases} cases")

test()


# def test_operacion(num_test_bin16, operando):
#     for i in range(len(num_test_bin16)):
#         for j in range(len(num_test_bin16)):
#             result_16bits = operando(num_test_bin16[i],num_test_bin16[j])
#             print("{}, {} = {}".format(num_test[0][i], num_test[0][j], result_16bits))
# mul = lambda a, b : a*b
# div = lambda a, b : a/b
# equ = lambda a, b : a==b
# no_equ = lambda a, b : a!=b
#test_operacion(num_test_bin16, mul)
#test_operacion(num_test_bin16, div)
#test_operacion(num_test_bin16, equ)
#test_operacion(num_test_bin16, no_equ)
