#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys

fh = open(sys.argv[3], "w") #Crear archivo para guardar los errores

inputLayerSize = 5
outputLayerSize = 1
hiddenLayerSize = 4

def sigmoid(x, deriv=False):
    if(deriv ==True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
    
X = np.loadtxt(sys.argv[1], delimiter = '\t') #Abrir el dataset de entrenamiento
y = np.loadtxt(sys.argv[2], dtype=int).reshape((-1, 1)) #Abrir el dataset de las salidas respectivas a las entradas de entrenamiento

np.random.seed(1)
#Pesos aleatorios entre [-1,1]
W1 = 2 * np.random.random((inputLayerSize,hiddenLayerSize)) - 1
W2 = 2 * np.random.random((hiddenLayerSize,outputLayerSize)) - 1 

np.savetxt('W1_iniciales.txt', W1, delimiter = '\t', fmt="%s")
np.savetxt('W2_iniciales.txt', W2, delimiter = '\t', fmt="%s")

for x in xrange(1000):
    #Propogate inputs though network
    l0 = X
    l1 = sigmoid(np.dot(l0,W1))
    l2 = sigmoid(np.dot(l1,W2))

    #Error de la salida
    l2_error = y - l2

    fh.write("Error: ")
    fh.write(str(np.mean(np.abs(l2_error))) + "\n")

    l2_delta = l2_error * sigmoid(l2,deriv=True)
    l1_error = l2_delta.dot(W2.T)
        
    l1_delta = l1_error * sigmoid(l1,deriv=True)
    W2 += l1.T.dot(l2_delta)
    W1 += l0.T.dot(l1_delta)

np.savetxt('W1_finales.txt', W1, delimiter = '\t', fmt="%s")
np.savetxt('W2_finales.txt', W2, delimiter = '\t', fmt="%s")
np.savetxt('Resultados.txt', l2, delimiter = '\t', fmt="%s")