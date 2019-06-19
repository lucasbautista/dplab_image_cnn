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
import cv2
import f_fimg as fim
import f_generator as fg
import os

DATASET_PATH  = 'data'
IMAGE_SIZE    = (500, 500)
CROP_LENGTH   = 125
NUM_CLASSES   = 2
BATCH_SIZE    = 1  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 20
WEIGHTS_FINAL = 'model-cropped-final.h5'
OUTPUT_SIZE = 500


    

def op_imgs(im,tot_op):
    
    (h,w) = im.shape[:2]
    tensor_output_size = (2*tot_op)+1
    tot_rot = tot_op
    Tensor_output = []
    Tensor_output.append(im) #salvo la original
    
    for i in range(1,tot_rot + 1):
        Tensor_output.append(fim.rotate_no_pad(im,angle = np.random.randint(0,360)))
    for j in range(tot_rot + 1,tensor_output_size):
        x_dis = np.random.randint(-w//2,w//2)
        y_dis = np.random.randint(-h//2,h//2)
        temp = fim.square_images(fim.translation_no_pad(im,tx=x_dis,ty=y_dis,orig_size=False),output_size=h)
        for k in range(0,temp.shape[0]):
            Tensor_output.append(temp[k])
    
    return np.stack(Tensor_output)
    #para mañana hay que agregar el codigo de traslacion a las imagenes, pensar cuanto desplazar o al menos dejarlo generico


def augmented_generator(batches, rows,cols,augmented_factor):
    """toma como entrada un iterador de keras de image data generator y aumenta el data set.
    particiona cada imagen generando que por cada imagen se genere rows*col imagenes, luego a cada particion
    se le aplican tecnicas de aumentacion, estandar, rotacion,ineccion de ruido,traslacion,etc.
    
    augmented factor, habla de cuantas imagenes quiero sacar a partir de una, la particion es independiente, augmented factor,
    tiene que ver con la cantidad de rotaciones aleatorias a realizar
    
    el resultado final es un batch de tamaño tot_batch = batch_size*((tot_imgs*augmented_factor) +augmented_factor)
    """
    while True:
        print("Ajustando imagenes al tamaño especificado")
        batch_x, batch_y = batches.my_next()
        batch_y = 'lateralizado'
        output = []
        batch_size = len(batch_x)
        for i_batch in range(0,batch_size):
            print("procesando________________________________ batch %s/%s"%(i_batch+1,batch_size))
            #esferizo las imagenes  obtengo un tensor de imagenes de resultado
            tensor_sq= fim.square_images(fim.chf_to_chl(batch_x[i_batch]),output_size=OUTPUT_SIZE)
            
            #llevar el tensor esferizado a una dimension de trabajo, con esa dimension se trabaja
            #implementar la funcion de optimal_resize_square_image la misma debe recibir una imagen
            # y decidir que metodo de resize utiliza en funcion de las dimensiones de la misma, luego
            #debe devolver dicha imagen
            
            
            #trabajo sobre el nuevo tensor,seteo sus parametros
            tensor_size = tensor_sq.shape[0]
            tensor_width = tensor_sq.shape[2]
            tensor_heigth = tensor_sq.shape[1]
            #estoy perdiendo la imagen original,el problema de la imagen original es que tiene distinto tamaño la tengo que reajustar
            tot_imgs = rows*cols
            tot_op = augmented_factor + 1 #cantidad de rotaciones y la original
        
            if(1 == tot_imgs):
                tot_tensor = tensor_size*tot_imgs*tot_op
            else:
                tot_tensor = tensor_size*((tot_imgs*tot_op) +tot_op)
            
            M = tensor_heigth//rows
            N = tensor_width//cols
            iter_n=0
            #triple for entre total_batch,x e y posicion de grid crop
            for n_tensor in range(0,tensor_size):
                o_img = tensor_sq[n_tensor]
                o_imgs = op_imgs(o_img,tot_op=augmented_factor)
                for i_op in range(0,o_imgs.shape[0]):
                    #aca hay que ver si tienen el tamaño de salida ver si es mas grande o mas chico y asi resize
                    #implementar una funcion que evalue dicha condicion y decida cual metodo usar mas optimo
                    #tensor_new[iter_n] = fim.chl_to_chf(cv2.resize(o_imgs[i_op],(500,500)))
                    output.append(fim.chl_to_chf(fim.optimal_resize_square_images(o_imgs[i_op],output_size=OUTPUT_SIZE)))
                    iter_n+=1
                
                if(1!=tot_imgs): # quiere decir que se decide no hacer particion
                    for y in range(0,tensor_heigth,M): #se lee, desde y hasta batch_heigth a pasos de M
                        for x in range(0, tensor_width, N):
                            tiles = tensor_sq[n_tensor,y:y+M,x:x+N,:] #salvo la seccion de imagen canales,filas,columnas
                            if  M==tiles.shape[0] and N==tiles.shape[1]:
                                crop_imgs = op_imgs(tiles,tot_op=augmented_factor)
                                for i_op in range(0,crop_imgs.shape[0]):
                                    #tensor_new[iter_n] = fim.chl_to_chf(cv2.resize(crop_imgs[i_op],IMAGE_SIZE))
                                    output.append(fim.chl_to_chf(fim.optimal_resize_square_images(crop_imgs[i_op],output_size=OUTPUT_SIZE)))
                                    iter_n+=1
            #output.append(tensor_new)
        batch_new = np.stack(output)
        print("se ajusto exitosamente")
        yield (batch_new, batch_y) #devuelve el resultado de la iteracion


train_datagen = fg.ImageDataGenerator(data_format='channels_first')
i = 0
#esto despues va a el generador
#train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
#                                                  target_size=(None,None),
#                                                  interpolation='bicubic',
#                                                  classes=['20um'], #que busque en el directorio de 20 um
#                                                  class_mode= 'categorical',
#                                                  shuffle=False,
#                                                  batch_size=BATCH_SIZE,
#                                                  save_to_dir= DATASET_PATH + '/temp_orig')
algo = 0
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=(None,None),
                                                  interpolation='bicubic',
                                                  classes=['20um'], #que busque en el directorio de 20 um
                                                  class_mode= 'categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  save_to_dir= DATASET_PATH + '/temp_orig')


valid_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=(None,None),
                                                  interpolation='bicubic',
                                                  classes=['20um'], #que busque en el directorio de 20 um
                                                  class_mode= 'categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  save_to_dir= DATASET_PATH + '/temp_orig')


#    print("algo")
#    algo +=1
#    if algo >5:
#        break


total_files = np.size(train_batches.filenames)
#seccion de debug

#First check 1 batch of uncropped images
#next y my_nex llaman al proximo elemento de batches, cuando llega al final saca el remanente, si se vuelve
#a llamar vuelve a empezar

rows = 2 
cols = 2
train_augmented = augmented_generator(train_batches, rows = rows,cols = cols,augmented_factor=5)

for i in range(0,total_files,BATCH_SIZE):
    batch_x, batch_y = next(train_augmented)
    batch_size_augmented =  len(batch_x)
    for j in range(0,batch_size_augmented):
        cv2.imwrite(os.path.join(DATASET_PATH + '/temp_orig' , '%s_%s_augmented.jpg'%(i,j)), fim.chf_to_chl(batch_x[j])[:,:,::-1])
        cv2.waitKey(0)
    #del batch_x, batch_y

## build our classifier model based on pre-trained ResNet50:
## 1. we don't include the top (fully connected) layers of ResNet50
## 2. we add a DropOut layer followed by a Dense (fully connected)
##    layer which generates softmax class score for each class
## 3. we compile the final model using an Adam optimizer, with a
##    low learning rate (since we are 'fine-tuning')
#net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
#               input_shape=(CROP_LENGTH,CROP_LENGTH,3))
#x = net.output
#x = Flatten()(x)
#x = Dropout(0.5)(x)
#output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
#net_final = Model(inputs=net.input, outputs=output_layer)
#for layer in net_final.layers[:FREEZE_LAYERS]:
#    layer.trainable = False
#for layer in net_final.layers[FREEZE_LAYERS:]:
#    layer.trainable = True
#net_final.compile(optimizer=Adam(lr=1e-5),
#                  loss='categorical_crossentropy', metrics=['accuracy'])
#print(net_final.summary())
#
## train the model
#net_final.fit_generator(train_augmented,
#                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
#                        validation_data = valid_crops,
#                        validation_steps = valid_batches.samples // BATCH_SIZE,
#                        epochs = NUM_EPOCHS)
#
## save trained weights
#net_final.save(WEIGHTS_FINAL)


