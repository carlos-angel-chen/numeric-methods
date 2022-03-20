#include "fizzbuzz.h"
#include <stdio.h>
#include <string.h>

#define MAX_VAL 999999999

/* fizzbuzz():
Recibe el numero de linea.
Devuelve un puntero a un string que contiene Fizz si es
multiplo de 3, Buzz si es multiplo de 5, FizzBuzz si es
multiplo de 3 y 5 a la vez, y el número para otros
naturales.
Si el numero es menor o igual a cero, o mayor a MAX_VAL,
devuelve un puntero nulo.
*/

char * fizzbuzz(int n)
{
   //Almaceno el string a devolver en una variable estatica,
   //para que no se pierda al salir de la funcion:
   static char output[10];

   if(n<=0 || n > MAX_VAL)
   {
      return NULL;
   }
   
   if(n%3 == 0 && n%5 == 0)
   {
      strcpy(output,"FizzBuzz");
   }
   else if(n%3 == 0)
   {
      strcpy(output,"Fizz");
   }
   else if(n%5 == 0)
   {
      strcpy(output,"Buzz");
   }
   else
   {
      sprintf(output,"%d",n);
   }

   return output;
}

