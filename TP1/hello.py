from floating_ieee754 import binary16

#pepe = binary16(2)
#pepa = binary16(4)

#blas = pepe + pepa
#blas.print()
#print(type(blas.float_num))


#pepe = binary16(2)
#papa = binary16(4)

#pepe*=papa

#print(type(pepe))

#print(type(float('nan') * 0))

#PARA PENSAR: como definimos /0, revisar IEEE

from floating_ieee754 import float_to_list, list_to_float

#lal = float_to_list(2)

#lel = float_to_list(2)

#lal[1] = 70
#print(lel)


#list_to_float([1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0])
#print(float_to_list(-5E-8)) #Nota: no "redondea", me temo que trunca

print('rta = ',list_to_float(float_to_list(-0.0)))