# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:52:05 2019
@author: Cynthia Callisaya 
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import pandas as pd
import math

#Parametros iniciales para la funcion de ajuste senoidal
p0 = [0.5, 60.0, -0.1]

df = pd.read_csv("datos/C6-PB-60.txt", sep="\t", names=['x','Intensidad'],header=0)

xs =df.iloc[:,0]/1.0
datos_y=df.iloc[:,1]

#nuevas variables para generar para las graficas
y1=np.random.rand(len(df))*0.0+np.mean(datos_y)

#funcion seno
def residuos(p, y, x):
    A, k, theta = p
    error = y - (A*np.sin(np.pi*x/k + theta)+np.mean(datos_y))
    return error
#Ajuste por minimos cuadrados y luego define una funcion para graficar con los parametros calculados
ajuste = leastsq(residuos, p0, args=(datos_y, xs))

def funcion(x, p):
    return p[0]*np.sin(np.pi*x/p[1] + p[2])+np.mean(datos_y)

# genero datos a partir del modelo para representarlo
y2 = funcion(xs, ajuste[0])# valor de la funcion modelo en los x
A,x,d=ajuste[0]#guardando los parametros de ajuste de la tuple para calculos futuros
plt.plot(xs, y2, 'b-')
plt.plot(xs, datos_y, 'k.')
plt.plot(xs, y1, 'r-')

#Interseccion y sus promedios para picos pares e impares
idx=np.argwhere(np.diff(np.sign(y1 - datos_y)) != 0).reshape(-1) + 0
xa=0.0
vec1 = []
vec2 = []

for i in range(len(idx)):
    plt.plot((xs[idx[i]]+xs[idx[i]+1])/2.,(y1[idx[i]]+y1[idx[i]+1])/2., 'ro')
    In=(xs[idx[i]]+xs[idx[i]+1])/2.
    resta=In-xa
    xa=In
    if i>0:
        if i%2 == 0:
            resta_par=resta
            vec1.append(resta_par)
        else:
            resta_impar=resta
            vec2.append(resta_impar)#guardando los valores en un vector

#plt.axhline(0, color='black')
plt.legend(['Funcion seno','Datos','Intensidad promedio','Interseccion'], loc='upper right')
plt.xlabel('longitud [um]')
plt.ylabel('Intensidad luz [UA]')
plt.savefig("plotInterseccionSeno.png", dpi=1000)
plt.show()

#salida de datos
print('x_par=',np.mean(vec1))
print('e_par=',np.std(vec1)/math.sqrt(len(vec1)-1))
print('x_impar=',np.mean(vec2))
print('e_impar=',np.std(vec2)/math.sqrt(len(vec2)-1))
print('Amplitud','Ancho','desfase [rad]')
print(ajuste[0])
print('Intensidad min','contraste (%)')
Imin=abs(np.mean(datos_y))-abs(A)
C=abs(A)*100/np.mean(datos_y)
print(Imin,C)