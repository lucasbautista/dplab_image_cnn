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

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] // 2), int(image_size[1] // 2))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - (width //2))
    x2 = int(x1 + width)
    y1 = int(image_center[1] - (height // 2))
    y2 = int(y1 + height)

    return image[y1:y2, x1:x2,:]

def rotate_around_center(image,angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def grid_crop(im,out_size = (250,250),dy=0,dx=0,angle = 45):
    "la funcion toma el primer elemento de la matriz y desplaza en las direcciones x,y a su vez rota" 
    (h,w) = im.shape[0:2]
    R = (dy,dx) # valor random donde cae la imagen
    #calculo el tamaño de grilla a tomar en funcion de la rotacion para tener 
    temp_angle = angle
    o = out_size[0] #de momento el algortimo solo funciona en imagenes cuadradas
    #la unica solucion que me sirve es la de los primeros 45 grados
    #luego se repite, mapeo a la entrada con mi solucion efectiva
    if(angle>45):
        temp_angle = -angle + 90
    if(angle>90):
        temp_angle = -angle + 135
    if(angle>=135):
        temp_angle = -angle + 180
    if(angle>180):
        temp_angle = -angle + 225
    if(angle>=225):
        temp_angle = -angle + 270
    if(angle>270):
        temp_angle = -angle + 315
    if(angle>=315):
        temp_angle = -angle + 360
    #calculo el angulo que representa la cantidad de pixeles a extraer en funcion de dicha rotacion
    angle_rad = (temp_angle*np.pi)/180
    
    if(temp_angle==0 or temp_angle==90 or temp_angle==180 or temp_angle == 270 or temp_angle ==360):
        n = o
    else:
        #calculo el tamaño de grilla en funcion de dicha rotacion empleada
        n = np.floor(abs(-(o*2*np.cos(angle_rad)*np.sin(angle_rad))/(1 - (np.cos(angle_rad) + np.sin(angle_rad))))).astype(int)
    
    
    #defino el nuevo tamaño de grilla para obtener la imagen de 0x0
    grid_size = (n,n) #tienen que quedar en funcion del angulo
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
    output = crop_around_center(rotate_around_center(im_t,angle),o,o)
    
    if(output.shape[0] != out_size[0] or output.shape[1] != out_size[1]):
        sys.exit("Algo fallo:\n tamaño imagen salida (%s,%s)\n los valores de dx y dy recibidos son (%s,%s)\n se roto %sº" 
                 %(output.shape[0],output.shape[1],R[1],R[0],angle))
    else:
        return output
        

def augmented_data(im,out_size = (250,250),k = 1):
    
    tensor_output = np.ones((k,out_size[0],out_size[1],3)).astype(np.uint8)
    angles = np.ones(k).astype(np.uint)
    origins = np.ones((2,k)).astype(np.uint)
    (h,w) = im.shape[0:2]
    angle = 0
    dx = 0
    dy = 0
    for i in range(0,k):
        angle = int(np.random.uniform(0,360)) #despues cuantizar
        angle = i
        angles[i] = angle
        dx = int(np.random.uniform(0,w))
        dy = int(np.random.uniform(0,h))
        origins[0][i] = dy
        origins[1][i] = dx
        tensor_output[i] = grid_crop(im,out_size=out_size,dy=dy,dx=dx,angle=angle)
    else:
        return tensor_output,angles,origins
        

pat_desp_image1 = cv2.imread("patron_desp.png")
pat_desp_image2 = cv2.imread("hrect.jpg")

pat_desp = pat_desp_image1

(h,w) = pat_desp.shape[0:2]
tensor,angles,origins = augmented_data(pat_desp,out_size=(279,279),k=360)


plt.figure(figsize=(10,10))
plt.imshow(pat_desp[:,:,::-1]),plt.title("Patron  %sX%s"%(pat_desp.shape[0:2]));

#descomentar y ejecutar para salvar todas las imagenes en una carpeta temp_oriog
for i in range(0,tensor.shape[0]):
    j = i+1
    cv2.imwrite(os.path.join('temp_orig' , '%s_augmented_rot_%sº_orig_(y,x)=(%s,%s).jpg'%
                             (j,angles[i],origins[0][i],origins[1][i])),tensor[i])





