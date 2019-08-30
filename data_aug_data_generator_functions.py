#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:27:49 2019

@author: lucas

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/


"""
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import cv2
import f_fimg as fim
import f_generator as fg
import os
import sys
import matplotlib.pyplot as plt

DATASET_PATH  = 'data'
IMAGE_SIZE    = (500, 500)
CROP_LENGTH   = 125
NUM_CLASSES   = 2
BATCH_SIZE    = 3  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 20
WEIGHTS_FINAL = 'model-cropped-final.h5'
OUTPUT_SIZE = 500

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


def get_grid_rot_square_size(output_size = 250,phi = 45):
    #calculo el tamaño de grilla a tomar en funcion de la rotacion para tener 
    theta = phi
    o = output_size #de momento el algortimo solo funciona en imagenes cuadradas    
    if(theta==0 or theta==90 or theta==180 or theta == 270 or theta ==360):
        n = o
    else:
    #calculo el tamaño de grilla en funcion de dicha rotacion empleada
    #la unica solucion que me sirve es la de los primeros 45 grados
    #luego se repite, mapeo a la entrada con mi solucion efectiva
        if(phi>45):
            theta = -phi + 90
        if(phi>90):
            theta = -phi + 135
        if(phi>=135):
            theta = -phi + 180
        if(phi>180):
            theta = -phi + 225
        if(phi>=225):
            theta = -phi + 270
        if(phi>270):
            theta = -phi + 315
        if(phi>=315):
            theta = -phi + 360
    #calculo el angulo que representa la cantidad de pixeles a extraer en funcion de dicha rotacion
        angle_rad = (theta*np.pi)/180
        n = np.floor(abs(-(o*2*np.cos(angle_rad)*np.sin(angle_rad))/(1 - (np.cos(angle_rad) + np.sin(angle_rad))))).astype(int)
    return n

def grid_crop(im,out_size = (250,250),dy=0,dx=0,angle = 45):
    "la funcion toma el primer elemento de la matriz y desplaza en las direcciones x,y a su vez rota" 
    (h,w) = im.shape[0:2]
    R = (dy,dx) # valor random donde cae la imagen
    n = get_grid_rot_square_size(output_size=out_size[0],phi=angle)
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
    output = crop_around_center(rotate_around_center(im_t,angle),out_size[0],out_size[1])
    
    if(output.shape[0] != out_size[0] or output.shape[1] != out_size[1]):
        plt.figure()
        plt.imshow(im[:,:,::-1])
        sys.exit("Algo fallo:\n tamaño imagen salida (%s,%s)\n los valores de dx y dy recibidos son (%s,%s)\n se roto %sº\n el n fue de %s\n" 
                 %(output.shape[0],output.shape[1],R[1],R[0],angle,n))
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
        angles[i] = angle
        dx = int(np.random.uniform(0,w))
        dy = int(np.random.uniform(0,h))
        origins[0][i] = dy
        origins[1][i] = dx
        tensor_output[i] = grid_crop(im,out_size=out_size,dy=dy,dx=dx,angle=angle)
    else:
        return tensor_output,angles,origins





def augmented_generator(batches,augmented_factor,out_size=(250,250)):
    """toma como entrada un iterador de keras de image data generator y aumenta el data set.
    particiona cada imagen generando que por cada imagen se genere aaugmented_factor imagenes, donde cada particion
    es el resultado de realizar un desplazamiento y una rotacion aleatoria a lo largo de la imagen horizontal.
    
    augmented factor, habla de cuantas imagenes quiero sacar a partir de una, la particion es independiente, augmented factor,
    tiene que ver con la cantidad de rotaciones y desplazamientos uniformes a realizar
    
    el resultado final es un batch de tamaño tot_batch = batch_size*augmented_factor para las muestrasç
    para las etiquetas es un batch de mismo tamaño con un vector de etiquetas en cada elemento
    """
    while True:
        batch_x, batch_y = batches.my_next()
        batch_y = 'lateralizado'
        batch_size = len(batch_x)
        tot_data = batch_size*augmented_factor
        output = (np.ones((tot_data,out_size[0],out_size[1],3))*0).astype(np.uint8)
        batch_y = (np.ones((tot_data,out_size[0],out_size[1],3))*0).astype(np.uint8)
        i_output = 0
        for i_batch in range(0,batch_size):
            #En funcion de h o w si estos son mas chicos que n_critic resampleo la imagen
            #el n critico ocurre para un tamaño de 250x250
            
            #esferizo la imagen y aumnento la cantidad de muestras de mi dataset a partir de una muestra
            tensor_sq,angles,origins= augmented_data(fim.chf_to_chl(batch_x[i_batch]),
                                      out_size=out_size,
                                      k = augmented_factor)
            
            for i_tensor_sq in range(0,tensor_sq.shape[0]):
                if i_output<output.shape[0]:
                    output[i_output] = tensor_sq[i_tensor_sq]
                    i_output = i_output + 1
        
        yield (output,batch_y) #devuelve el resultado de la iteracion


train_datagen = fg.ImageDataGenerator(data_format='channels_first')
i = 0

#esto hay que reemplazarlo junto con augmented generator
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=(None,None),
                                                  classes=['50um','20um'], #que busque en el directorio de 20 um
                                                  class_mode= 'categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  save_to_dir= DATASET_PATH + '/temp_orig')


total_files = np.size(train_batches.filenames)
#seccion de debug
k=0
train_augmented = augmented_generator(train_batches,augmented_factor=10)
i_file = 0
for i in range(0,total_files):
    print("Augmentando:________________________________ archivo %s/%s"%(i+1,total_files))
    if k*BATCH_SIZE == i:
        k = k + 1
        batch_x, batch_y = next(train_augmented)
        batch_size_augmented =  len(batch_x)
        for j in range(0,batch_size_augmented):
            cv2.imwrite(os.path.join(DATASET_PATH + '/temp_orig' , '%s_%s_augmented.jpg'%(i,j)),batch_x[j][:,:,::-1])
            cv2.waitKey(0)
    #del batch_x, batch_y
print("fin augmentacion\n")













