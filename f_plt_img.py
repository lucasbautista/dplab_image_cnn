#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:35:18 2019

@author: lucas
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def bgr_to_rgb(bgr):
    rgb = bgr[...,::-1]
    return rgb

def rgb_to_bgr(rgb):
    bgr = rgb[...,::-1]
    return bgr

def plot_bgr(img):
    plt.imshow(bgr_to_rgb(img))

def plot_rgb(img):
    plt.imshow(img)

def plot_gray(img):
    # img could be either (H,W) or (H,W,C)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

def batch_plots(ims, is_bgr=True, figsize=(12,6), rows=1, interp=False, titles=None):
    ndims = len(ims[0].shape)
    assert ndims == 3 or ndims == 2
    if ndims == 2:  # convert grayscale images to bgr
        ims = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in ims]
    ims = np.array(ims).astype(np.uint8)
    if (ims.shape[-1] != 3):
        ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        if is_bgr:
            plt.imshow(bgr_to_rgb(ims[i]), interpolation=None if interp else 'none')
        else:
            plt.imshow(ims[i], interpolation=None if interp else 'none')
            

def plots(ims, is_bgr=True, figsize=(12,6), rows=1, cols=1,interp=False, titles=None):
    ndims = len(ims[0].shape)
    assert ndims == 3 or ndims == 2
    if ndims == 2:  # convert grayscale images to bgr
        ims = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in ims]
    ims = np.array(ims).astype(np.uint8)
    if (ims.shape[-1] != 3):
        ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        if i<rows*cols:
            sp = f.add_subplot(rows, cols, i+1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            if is_bgr:
                plt.imshow(bgr_to_rgb(ims[i]), interpolation=None if interp else 'none')
            else:
                plt.imshow(ims[i], interpolation=None if interp else 'none')
                
def list_plots(ims, is_bgr=True, figsize=(12,6), rows=1, cols=1,interp=False, titles=None):
    ndims = len(ims[0].shape)
    assert ndims == 3 or ndims == 2
    if ndims == 2:  # convert grayscale images to bgr
        ims = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in ims]
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        if i<rows*cols:
            ploting = np.array(ims[i]).astype(np.uint8)
            if (ploting.shape[0] == 3):
                ploting = ploting.transpose((1,2,0))
            sp = f.add_subplot(rows, cols, i+1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            if is_bgr:
                plt.imshow(bgr_to_rgb(ploting), interpolation=None if interp else 'none')
            else:
                plt.imshow(ploting, interpolation=None if interp else 'none')
                
                
                
                
                