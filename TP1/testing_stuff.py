print("Oh Yeah Mr. Krabs")

import struct
import binascii
from floating_ieee754 import binary16
import numpy as np

# print(binascii.hexlify(struct.pack('<d',+0)))

# print(binascii.hexlify(struct.pack('<d',float('inf'))))

# print(binascii.hexlify(struct.pack('<d',float('-inf'))))

# print(binascii.hexlify(struct.pack('<d',float('nan'))))

# print(binascii.hexlify(struct.pack('<d',+0)))

# print(binascii.hexlify(struct.pack('<d',+0.0)))

# print(binascii.hexlify(struct.pack('<d',-0.0)))

a = 5.984634
b = 2.434947
a_16 = np.float16(a)
b_16 = np.float16(b)

pepe = binary16(a)
papa = binary16(b)



# def bool2num(bits):
#     num = []
#     for i in range(len(bits)):
#         if bits[i]==False:
#             num.append(0)
#         else:
#             num.append(1)
#     return num

# print(bool2num(pepe.bits))
# print(pepe.bits)
# print(papa.bits)

# print((pepe*papa).d)
# print(a_16*b_16)
# print(np.half(a*b))

x = binary16(0.0)
y = binary16(-0.0)

print(x.bits)
print(y.bits)

"""
TESTING

list = [16 bits]
d = entero de 64 bits

float_to_list()
    1. convertir float to list
    2. verificar que la list de 1 y 0 coincida con el float asociado a normal
    3. verificar que la list de 1 y 0 coincida con el float asociado a subnormal
    4. verificar que la list tenga los 1 y 0 para nan
    5. verificar que la list tenga los 1 y 0 para +inf 
    6. verificar que la list tenga los 1 y 0 para -inf
    7. verificar que la list tenga los 1 y 0 para +0 (nota: escribir +0.0)
    8. verificar que la list tenga los 1 y 0 para -0 (nota: escribir -0.0)

list_to_float()
    1. convertir list to float
    2. verificar que el float concida con los 1 y 0 del list asociado a normal
    3. verificar que el float concida con los 1 y 0 del list asociado a subnormal
    4. verificar que la list nan convertida a float sea nan (nota: usar isnan())
    5. verificar que la list +inf convertida a float sea +inf 
    6. verificar que la list -inf convertida a float sea -inf 
    7. verificar que la list +0 convertida a float sea +0 (nota: escribir +0.0)
    8. verificar que la list -0 convertida a float sea -0 (nota: escribir -0.0)

__mul__()
    1. float * float (normal y subnormal)
    2. int * float (normal y subnormal)
    3. inf * inf
    4. -inf * inf
    5. -inf * -inf
    6. 0 * 0
    7. -0 * 0 
    8. -0 * -0 
    9. nan * nan
    10. inf * 0
    11. inf * -0 
    12. -inf * 0
    13. -inf * -0
    14. 

__truediv__()
    nota: analizar los 36 casos si o si (FUCK)
"""

