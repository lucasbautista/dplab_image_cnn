#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:38:32 2019

@author: lucas
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

image = cv2.imread("patron_desp.png")
plt.figure(figsize=(10,10))
plt.imshow(image[:,:,::-1]),plt.title("Patron Particionado %sX%s"%(image.shape[0:2]));

##################codigo######################################
tiles = []
(h,w) = image.shape[0:2]
out_size = 250
j=0
flag_plot = 1
k = 0
for y in range(0,h,out_size):
    k = k+1
    print("%s\n"%(k))
    for x in range(0, w, out_size):
        y1 = y + out_size
        x1 = x + out_size
        n_img = image[y:y1,x:x1,:]
        if n_img.shape[:2] == (out_size,out_size):
                tiles.append(n_img)
tiles = np.stack(tiles) #convierto a tensor
#descomentar y ejecutar para salvar todas las imagenes en una carpeta temp_oriog
#for i in range(0,tiles.shape[0]):
#    j = i+1
#    cv2.imwrite(os.path.join('temp_orig' , '%s_croped.jpg'%j), tiles[i])

lista_imagenes = (np.ones(tiles.shape[0])*0).astype(int)
for i in range(0,tiles.shape[0]):
    lista_imagenes[i]=i+1


labels = np.array(lista_imagenes)
print("Recuerde que las Imagenes se Listan del Numero 1 al %s\n"%tiles.shape[0])
for i in range(0,labels.shape[0]):
    if(labels[i]<=tiles.shape[0]) and labels[i]>0:
        if flag_plot:
            plt.figure(figsize=(30,5))
            flag_plot = 0
        if(j<8):
            plt.subplot(1,8,j+1),plt.imshow(tiles[labels[i]-1][:,:,::-1])
            j = j+1
            if(j>=8):
                flag_plot = 1
                j=0
    else:   
        print("La Imagen numero %s que Usted Selecciono se Encuentra Fuera de Rango,Dispone de %s Imagenes\n"%
                (labels[i],tiles.shape[0]))