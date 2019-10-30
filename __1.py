"""
Błąd punktowy zbioru danych.

Punkty w zbiorze danych są poprawne w zależności od ich otoczenia.
Punkt, którego pochodna dyskretna jest wysoka, wykazuje mniejszą poprawność,
niż te o niskiej pochodnej dyskretnej. 
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import DataStruct
from __Main import inte, wave


Wavenumber = wave[0]
Intensity = inte[0]

dI = DataStruct.deriv(Wavenumber, Intensity)

plt.plot(Wavenumber, dI, '.')
plt.show()

# def primes(number):
#     prs = []
#     value_range = number
#     while number > 1:
#         for n in range(2, value_range+1):
#             if number % n == 0:
#                 number = int(number/n)
#                 prs.append(n)
#                 break
#     return prs


# def polynom(zeros):
#     import math

#     def func(X):
#         result = []
#         for x in X:
#             value = 1
#             for zero in zeros:
#                 value *= (x - zero)
#             result.append(value)
#         return result
#     return func
