#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:27:33 2019

@author: lucas
"""

from __future__ import print_function
import numpy as np
import cv2
import math

def chf_to_chl(data): #convierte una imagen de formato, canal primero a canal ultimo
    #chf: ch,row,col
    #chl: row,col,ch:  
    return data.transpose((1,2,0))

def chl_to_chf(data): #convierte una imagen de formato canal ultimo a canal primero
    #chf: ch,row,col
    #chl: row,col,ch:  
    return data.transpose((2,0,1))

def optimal_resize_square_images(im,output_size):
    (h,w) = im.shape[0:2]
    if(output_size==h or output_size==w):
        return im
    #evaluo si la imagen cuadrada es mas grange o mas chica que la original
    if(h>output_size or w>output_size):
        return cv2.resize(im,(output_size,output_size),interpolation = cv2.INTER_AREA)
    else:
        return cv2.resize(im,(output_size,output_size),interpolation = cv2.INTER_CUBIC)

def optimal_resize_image_tensor(tensor,output_size):
    return np.stack([optimal_resize_square_images(tensor[i],output_size=output_size) 
                        for i in range(0,tensor.shape[0])])



def square_images(im,output_size):
    """
    Recibe una imagen en chl y devuelve un tensor de imagenes de NXN donde N es la menor dimension de la imagen
    e.g si se recibe una imagen de 540*1240 el resultado seran varias imagenes de 540*540, las cuales cubren
    todas las zonas de interes de la original.
    
    si la imagen es cuadrada se devuelve un tensor con la imagen original y una imagen en negro a su vez un indicador de que
    la imagen es cuadrada
    """
    imgheight=im.shape[0]
    imgwidth=im.shape[1]
    #busco si la imagen es ancha larga o cuadrada para saber en que direccion cortar
    if imgheight==imgwidth:
        tr = np.ones((1,im.shape[0],im.shape[1],im.shape[2]))*0
        tr[0] = im        
        return optimal_resize_image_tensor(tr,output_size=output_size)
    elif(imgheight<imgwidth):
        out_size = imgheight
    else:
        out_size = imgwidth
    
    tiles = []
    for y in range(0,imgheight,out_size):
        for x in range(0, imgwidth, out_size):
            y1 = y + out_size
            x1 = x + out_size
            n_img = im[y:y1,x:x1]
            if n_img.shape[:2] == (out_size,out_size):
                tiles.append(n_img)

    if(imgheight<imgwidth):
        for y in range(0,imgheight,out_size):
            for x in range(imgwidth, 0, -out_size):
                y1 = y + out_size
                x1 = x - out_size
                n_img = im[y:y1,x1:x]
                if n_img.shape[:2] == (out_size,out_size):
                    tiles.append(n_img)
    else:
        for y in range(imgheight,0,-out_size):
            for x in range(0, imgwidth, out_size):
                y1 = y - out_size
                x1 = x + out_size
                n_img = im[y1:y,x:x1]
                if n_img.shape[:2] == (out_size,out_size):
                    tiles.append(n_img)

    return optimal_resize_image_tensor(np.stack(tiles),output_size=output_size)

def image_auto_resize(image,factor,inter = cv2.INTER_AREA):
    """
    Factor: valor nuevo que se necesita, de altura o anchura
    la funcion, decide de que forma es mejor,,en terminos de area llevar la imagen
    al nuevo valor pedido, util para esferizar
    """
    (h, w) = image.shape[:2]
    wo = factor
    ho = factor
    #computo ratios de cambio de imagen
    rh = ho/h
    rw = wo/w  
    #impongo que mi nueva altura es la pedida
    h_prior = ho
    #afecto a al ancho por el ratio de crecimiento y evaluo si cumple ser mayor a la salida
    w_prior = int(w*rh)
    if(w_prior<wo):
        #fuerzo al ancho a ser el valor
        w_y1 = wo
        #al modificar el ancho debo remodificar el alto,recomputo el ratio y recalculo el ancho
        h_y1 = int(ho*(w_y1/w_prior ))
    else:
        w_y1 = w_prior
        h_y1 = h_prior
    #mismo procedimiento pero ahora inicializo con el ancho
    w_prior = wo
    h_prior = int(h*rw)
    if(h_prior<ho):
        #fuerzo al ancho a ser el valor
        h_y2 = ho
        #al modificar el ancho debo remodificar el alto,recomputo el ratio y recalculo el ancho
        w_y2 = int(wo*(h_y2/h_prior ))
    else:
        w_y2 = w_prior
        h_y2 = h_prior

    if((w_y1*h_y1)<=(w_y2*h_y2)):
        w_y = w_y1
        h_y = h_y1
    else:
        w_y = w_y2
        h_y = h_y2
    
    return cv2.resize(image,(w_y,h_y),interpolation = inter)


def image_resize(image, n_width = None, n_height = None, inter = cv2.INTER_AREA):
    
    dim = None
    (h, w) = image.shape[:2]
    if n_width is None and n_height is None:
        return image
    if n_width is None:
        #cambia en la altura especificada y el ancho se ajusta para mantener proporcion
        r = n_height / float(h)
        dim = (int(w * r), n_height)
    # otherwise, the height is None
    else:
        #cambia en el ancho especificado y la altuira se ajusta para mantener proporcion
        r = n_width / float(w)
        dim = (n_width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

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
    x2 = int(image_center[0] + (width // 2))
    y1 = int(image_center[1] - (height // 2))
    y2 = int(image_center[1] + (height //2))

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


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))



def rotate_no_pad(image,angle):
#rota una imagen un angulo en grados y tras definir el mejor rectangulo realiza el crop desde ese punto    
    (h, w) = image.shape[:2]
    h_r,w_r = rotatedRectWithMaxArea(w, h, angle=(angle*np.pi)/180)
    #computo imagen rotada,cropeo en torno al mejor rectangulo
    return  cv2.resize(crop_around_center(rotate_bound(image,angle),w_r,h_r),(w,h))

def translation_no_pad(image,tx,ty,orig_size=False):
    """
    toma una imagen en formato channel last, y la desplaza tx pixeles en horizontal
    y tx pixeles en vertical,de ser necesario se devuleve a su tamaño original 
    si:
    tx>0 el desplazamiento es a derecha; tx<0 el desplazamiento es a izquierda
    ty>0 el desplazamiento es abajo; ty<0 el desplazamiento es hacia arriba
    """
    #guardo el tamaño original del la imagen
    (h, w) = image.shape[:2]
    #computo matriz traslacion
    M = np.float32([[1,0,tx],[0,1,ty]])
    #valores ansolutos de desplazamiento
    atx = abs(tx)
    aty = abs(ty)
    #inicializo los flags en un valor arbitrario
    stx = False
    sty = False
    #me fijo si es, un numero positivo o negativo
    if(tx>0):
        stx=False
    elif tx<0:
        stx=True
    if(ty>0):
        sty=False
    elif ty<0:
        sty=True
    #niego el valor de los flags
    nstx = not(stx)
    nsty = not(sty)
    #realizo la funcion de traslacion y luego cropping con logica booleana 
    output = cv2.warpAffine(image,M,(w,h))[nsty*aty:h-(aty*sty),nstx*atx:w-(atx*stx),:]
    #vuelvo al tamaño original si es que lo necesito
    if(orig_size):
        output = cv2.resize(output,(w,h))
    return output
    
