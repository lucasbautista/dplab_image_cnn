#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:03:44 2019

@author: lucas
"""
from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def grid_crop(im,out_size = (250,250),dy=0,dx=0,angle = 45):
    "la funcion toma el primer elemento de la matriz y desplaza en las direcciones x,y a su vez rota" 
    (h,w) = im.shape[0:2]
    R = (dy,dx) # valor random donde cae la imagen
    #aca va la conversion entre lo que quiero rotar y cuanto voy a desplazar para generar la imagen de salida de OXX
    grid_size = (out_size[0],out_size[1]) #tienen que quedar en funcion del angulo
    
    grid_size = (355,355) #borrar esto
    disp_x = grid_size[1]//2 #cuanto desplazo en X
    disp_y = grid_size[0]//2 #cuanto me desplazo en Y
    
    if(R[1] + grid_size[1] >= w): #caigo en una zona fuera de la imagen en x
        xi = w - grid_size[1]
        xf = w 
    else: #es un punto donde no hay ningun conflicto
        disp_x = R[1] + grid_size[1]
        xi = R[1]
        xf = disp_x
        
    if(R[0] + grid_size[0] >= h): #caigo en una zona fuera de la imagen en x
        yi = h - grid_size[0]
        yf = h
    else: #es un punto donde no hay ningun conflicto
        disp_y = R[0] + grid_size[0]
        yi = R[0]
        yf = disp_y
    
    #esto es para la traslacion, para la rotacion usamos
    
    im_t = im[yi:yf,xi:xf,:] #los rangos truncan y no permiten acceder a posiciones de memorias invalidas
    
    if(im_t.shape[0] != grid_size[0] or im_t.shape[1] != grid_size[1]):
        sys.exit("Algo fallo:\n tamaño imagen salida (%s,%s)\n los valores de dx y dy recibidos son (%s,%s)\n" 
                 %(im_t.shape[0],im_t.shape[1],R[1],R[0]))
    else:
        return im_t
        

def augmented_data(im,out_size = (250,250),k = 1):
    
    tensor_output = np.ones((k,out_size[0],out_size[1],3)).astype(np.uint8)
    (h,w) = im.shape[0:2]
    angle = 0
    dx = 0
    dy = 0
    for i in range(0,k):
        angle = int(np.random.uniform(0,360)) #despues cuantizar
        dx = int(np.random.uniform(0,w))
        dy = int(np.random.uniform(0,h))
        tensor_output[i] = grid_crop(im,out_size=(250,250),dy=dy,dx=dx,angle=angle)
    else:
        return tensor_output
        

pat_desp_image1 = cv2.imread("patron_desp.png")
pat_desp_image2 = cv2.imread("hrect.jpg")

pat_desp = pat_desp_image1

(h,w) = pat_desp.shape[0:2]
tensor = augmented_data(pat_desp,out_size=(355,355),k=2000)


plt.figure(figsize=(10,10))
plt.imshow(pat_desp[:,:,::-1]),plt.title("Patron  %sX%s"%(pat_desp.shape[0:2]));

#descomentar y ejecutar para salvar todas las imagenes en una carpeta temp_oriog
for i in range(0,tensor.shape[0]):
    j = i+1
    cv2.imwrite(os.path.join('temp_orig' , '%s_augmented.jpg'%j), tensor[i])





