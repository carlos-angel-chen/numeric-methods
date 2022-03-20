#include "fizzbuzz.h"
#include <stdio.h>
#include <string.h>

int pruebas_totales = 0;
int pruebas_pass = 0;
int pruebas_fail = 0;

void check_string_iguales(char * esperado, char * actual);
void check_punteros_iguales(char * esperado, char * actual);

int main(void)
{
   printf("Corriendo pruebas en modulo fizzbuzz.c...");

   check_punteros_iguales(NULL,fizzbuzz(0));
   check_punteros_iguales(NULL,fizzbuzz(-1));
   check_punteros_iguales(NULL,fizzbuzz(1000000000));

   check_string_iguales("1",fizzbuzz(1));
   check_string_iguales("2",fizzbuzz(2));
   check_string_iguales("11",fizzbuzz(11));
   check_string_iguales("299",fizzbuzz(299));

   check_string_iguales("Fizz",fizzbuzz(3));
   check_string_iguales("Fizz",fizzbuzz(12));
   check_string_iguales("Fizz",fizzbuzz(27));
   check_string_iguales("Fizz",fizzbuzz(303));

   check_string_iguales("Buzz",fizzbuzz(5));
   check_string_iguales("Buzz",fizzbuzz(20));
   check_string_iguales("Buzz",fizzbuzz(100));
   check_string_iguales("Buzz",fizzbuzz(500));

   check_string_iguales("FizzBuzz",fizzbuzz(15));
   check_string_iguales("FizzBuzz",fizzbuzz(90));
   check_string_iguales("FizzBuzz",fizzbuzz(300));
   check_string_iguales("FizzBuzz",fizzbuzz(1005));

   printf("\n%d pruebas realizadas: %d PASS, %d FAIL\n",pruebas_totales,pruebas_pass,pruebas_fail);

   return 0;
}

void check_string_iguales(char * esperado, char * actual)
{
   pruebas_totales++;

   if(strcmp(esperado,actual)==0)
   {
      pruebas_pass++;
   }
   else
   {
      pruebas_fail++;
      printf("\nFAIL - esperado: %s - actual: %s", esperado, actual);
   }
}

void check_punteros_iguales(char * esperado, char * actual)
{
   pruebas_totales++;

   if(esperado == actual)
   {
      pruebas_pass++;
   }
   else
   {
      pruebas_fail++;
      printf("\nFAIL - esperado: %p - actual: %p", esperado, actual);
   }
}
