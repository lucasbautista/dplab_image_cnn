#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:35:37 2019

@author: lucas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import cv2
import sys

#%% Funciones: Filtrado de String y normalizacion de datos
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def norm_data(data):
    return (data-data.mean())/data.std()

def open_image(path):
    image = cv2.imread(path + ".jpg")
    if (image is None):
        image = cv2.imread(path + ".png")
        if(image is None):
            image = cv2.imread(path + ".tif")
            if(image is None):
                sys.exit("%s___no se pudo read"%(path))
    return image

def clasificador(data,list_class,name_class):
    #quiero devolver un vector,con los datos en funcion de la clase
    output = []
    for i in range(0,data.size):
        if(list_class[i]==list_class[i]):
            if(name_class == list_class[i]):
                output.append(np.array([data[i]]))
    return output


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% Carga De Datos
df = pd.read_csv("data_celulas.csv")
n_df = df[['AREA_TT_OV_TOTAL','IOD_TT_OV_TOTAL']]
Area_TT_sobre_Total = np.array(df.AREA_TT_OV_TOTAL)
IOD_TT_total = np.array(df.IOD_TT_OV_TOTAL)
Classes = np.array(df.TTO)
snaps_id = np.array(df.SNAP);
paths = 'data' + '/train'+'/'+np.array(df.DIR)+'/'+snaps_id

#%% Filtrado de Datos
IOD_TT_total_filter = []
Area_TT_sobre_Total_filter = []
aty_paths_list = []
scatter = []
flag_1 = 0
flag_2 = 0

#filtro los datos que no esten completos extraigo los numeros y armo el vector para el scatter
for i in range(0,IOD_TT_total.size):
    #me fijo de que no sea un nan
    if(IOD_TT_total[i]==IOD_TT_total[i]):
        #si no es un nan ahora me fijo que corresponda a un numero
        if(is_number(IOD_TT_total[i])):
            #corresponde a un numero entonces lo guardo
            flag_1 = 1
            temp_n1 = float(IOD_TT_total[i])
            IOD_TT_total_filter.append([temp_n1,i,Classes[i],snaps_id[i],paths[i]])
    #me fijo de que no sea un nan
    if(Area_TT_sobre_Total[i]==Area_TT_sobre_Total[i]):
        #si no es un nan ahora me fijo que corresponda a un numero
        if(is_number(Area_TT_sobre_Total[i])):
            #corresponde a un numero entonces lo guardo
            flag_2 = 1
            temp_n2 = float(Area_TT_sobre_Total[i])
            Area_TT_sobre_Total_filter.append([temp_n2,i,Classes[i],snaps_id[i],paths[i]])
    
    if(flag_1!=0 and flag_2 !=0):
        #significa que ambos son numeros
        ##aca me tengo que guardar el directorio, cosa de ir a buscar los atipicos despues
        scatter.append(np.array([temp_n2,temp_n1,i,Classes[i],snaps_id[i],paths[i]]))
        aty_paths_list.append(paths[i])
    #reseteo los flags
    flag_1 = 0
    flag_2 = 0


IOD_TT_total_filter = np.array(IOD_TT_total_filter).transpose()
Area_TT_sobre_Total_filter = np.array(Area_TT_sobre_Total_filter).transpose()
scatter = np.array(scatter).transpose()

#%% me mquedo solo con los datos, para el procesamiento

IOD_TT_total_filter_data = np.array([IOD_TT_total_filter[0]]).astype(np.float64)[0]
Area_TT_sobre_Total_filter_data = np.array([Area_TT_sobre_Total_filter[0]]).astype(np.float64)[0]
scatter_data = np.array([scatter[0],scatter[1]]).astype(np.float64)

#mostrar algunos elementos de la tabla
df[['DIR','SNAP','AREA_TT_OV_TOTAL','IOD_TT_OV_TOTAL','perido exptal','TTO']]


#%% Clasificacion de clases

data_control =          (np.array(Area_TT_sobre_Total_filter[0]['control' == Area_TT_sobre_Total_filter[2]]).astype(np.float64),
                         np.array(Area_TT_sobre_Total_filter[1]['control' == Area_TT_sobre_Total_filter[2]]).astype(np.int),
                         Area_TT_sobre_Total_filter[3]['control' == Area_TT_sobre_Total_filter[2]],
                         Area_TT_sobre_Total_filter[4]['control' == Area_TT_sobre_Total_filter[2]])

data_MEL =              (np.array(Area_TT_sobre_Total_filter[0]['Mel' == Area_TT_sobre_Total_filter[2]]).astype(np.float64),
                        np.array(Area_TT_sobre_Total_filter[1]['Mel' == Area_TT_sobre_Total_filter[2]]).astype(np.int),
                        Area_TT_sobre_Total_filter[3]['Mel' == Area_TT_sobre_Total_filter[2]],
                        Area_TT_sobre_Total_filter[4]['Mel' == Area_TT_sobre_Total_filter[2]])

data_LUZ =              (np.array(Area_TT_sobre_Total_filter[0]['Luz' == Area_TT_sobre_Total_filter[2]]).astype(np.float64),
                        np.array(Area_TT_sobre_Total_filter[1]['Luz' == Area_TT_sobre_Total_filter[2]]).astype(np.int),
                        Area_TT_sobre_Total_filter[3]['Luz' == Area_TT_sobre_Total_filter[2]],
                        Area_TT_sobre_Total_filter[4]['Luz' == Area_TT_sobre_Total_filter[2]])

data_MELplusLUZ =      (np.array(Area_TT_sobre_Total_filter[0]['Mel+Luz' == Area_TT_sobre_Total_filter[2]]).astype(np.float64),
                        np.array(Area_TT_sobre_Total_filter[1]['Mel+Luz' == Area_TT_sobre_Total_filter[2]]).astype(np.int),
                        Area_TT_sobre_Total_filter[3]['Mel+Luz' == Area_TT_sobre_Total_filter[2]],
                        Area_TT_sobre_Total_filter[4]['Mel+Luz' == Area_TT_sobre_Total_filter[2]])

plt.figure(figsize=(30,20))
plt.rc('font', size=365)          # controls default text sizes
plt.rc('axes', titlesize=65)     # fontsize of the axes title
plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=32)    # fontsize of the tick labels
plt.rc('ytick', labelsize=32)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.style.use('seaborn-white')
plt.title('\nHistograma ATT\n')
plt.hist(data_control[0],bins=30,alpha = 0.3,histtype='stepfilled',color='blue',
         orientation='vertical',label='Control')
plt.hist(data_LUZ[0],bins=10,alpha = 0.8,histtype='bar',color = 'red',
        orientation='vertical',label='Luz')
plt.hist(data_MEL[0],bins=15,alpha = 0.3,histtype='stepfilled',color = 'green',
        orientation='vertical',label='Mel')
plt.hist(data_MELplusLUZ[0],bins=15,alpha = 0.8,histtype='bar',color = 'black',
         orientation='vertical',label='Mel + Luz')
plt.xlabel("$A_{TT}$")
plt.ylabel("Nº Muestras")
plt.legend()
plt.show()
#deuelvo los parametros por defecto de las figuras
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#separamos en funcion de su graddo de laterlacion

lat = 48; #todo valor por debajo de 48 esta lateralizado
no_lat = 65 #todo valor por encima de not_lat lo considero no_laterlaziado

data_control_notlat = (data_control[0][data_control[0]>no_lat],
                       data_control[1][data_control[0]>no_lat],
                       data_control[2][data_control[0]>no_lat],
                       data_control[3][data_control[0]>no_lat])

data_control_lat =      (data_control[0][data_control[0]<lat],
                         data_control[1][data_control[0]<lat],
                         data_control[2][data_control[0]<lat],
                         data_control[3][data_control[0]<lat])


data_MEL_notlat =       (data_MEL[0][data_MEL[0]>no_lat],
                         data_MEL[1][data_MEL[0]>no_lat],
                         data_MEL[2][data_MEL[0]>no_lat],
                         data_MEL[3][data_MEL[0]>no_lat])
                         

data_MEL_lat =          (data_MEL[0][data_MEL[0]<lat],
                         data_MEL[1][data_MEL[0]<lat],
                         data_MEL[2][data_MEL[0]<lat],
                         data_MEL[3][data_MEL[0]<lat])

data_LUZ_notlat =       (data_LUZ[0][data_LUZ[0]>no_lat],
                         data_LUZ[1][data_LUZ[0]>no_lat],
                         data_LUZ[2][data_LUZ[0]>no_lat],
                         data_LUZ[3][data_LUZ[0]>no_lat])

data_LUZ_lat =           (data_LUZ[0][data_LUZ[0]<lat],
                         data_LUZ[1][data_LUZ[0]<lat],
                         data_LUZ[2][data_LUZ[0]<lat],
                         data_LUZ[3][data_LUZ[0]<lat])

data_MELplusLUZ_notlat = (data_MELplusLUZ[0][data_MELplusLUZ[0]>no_lat],
                          data_MELplusLUZ[1][data_MELplusLUZ[0]>no_lat],
                          data_MELplusLUZ[2][data_MELplusLUZ[0]>no_lat],
                          data_MELplusLUZ[3][data_MELplusLUZ[0]>no_lat])

data_MELplusLUZ_lat =    (data_MELplusLUZ[0][data_MELplusLUZ[0]<lat],
                          data_MELplusLUZ[1][data_MELplusLUZ[0]<lat],
                          data_MELplusLUZ[2][data_MELplusLUZ[0]<lat],
                          data_MELplusLUZ[3][data_MELplusLUZ[0]<lat])


#separamon en funcion del caso extremo

#control
control_lat_x = (open_image(data_control[3][np.argmin(data_control[0])]), #imagen
                             data_control[2][np.argmin(data_control[0])], #snap_id
                             np.min(data_control[0]))                       #valor de ATT

control_notlat_x = (open_image(data_control[3][np.argmax(data_control[0])]),
                            data_control[2][np.argmax(data_control[0])],
                                np.max(data_control[0]))




MEL_lat_x = (open_image(data_MEL[3][np.argmin(data_MEL[0])]),
                         data_MEL[2][np.argmin(data_MEL[0])],
                         np.min(data_MEL[0]))

MEL_notlat_x = (open_image(data_MEL[3][np.argmax(data_MEL[0])]),
                        data_MEL[2][np.argmax(data_MEL[0])],
                                np.max(data_MEL[0]))




LUZ_lat_x = (open_image(data_LUZ[3][np.argmin(data_LUZ[0])]),
                         data_LUZ[2][np.argmin(data_LUZ[0])],
                         np.min(data_LUZ[0]))

LUZ_notlat_x = (open_image(data_LUZ[3][np.argmax(data_LUZ[0])]),
                        data_LUZ[2][np.argmax(data_LUZ[0])],
                                np.max(data_LUZ[0]))




MELplusLUZ_lat_x = (open_image(data_MELplusLUZ[3][np.argmin(data_MELplusLUZ[0])]),
                         data_MELplusLUZ[2][np.argmin(data_MELplusLUZ[0])],
                         np.min(data_MELplusLUZ[0]))

MELplusLUZ_notlat_x = (open_image(data_MELplusLUZ[3][np.argmax(data_MELplusLUZ[0])]),
                           data_MELplusLUZ[2][np.argmax(data_MELplusLUZ[0])],
                                np.max(data_MELplusLUZ[0]))


## graficamos muestras de casos cercanos a los umbrales propuestos,en caso de que no exista una muestra dentro dentro
#de ese umbral reemplazamos por su caso mas extremos

#control_lat
if np.size(data_control_lat)>0:
    control_lat = (open_image(data_control_lat[3][np.argmax(data_control_lat[0])]),
                   data_control_lat[2][np.argmax(data_control_lat[0])],
                   np.max(data_control_lat[0]))
else:
    print("Advertencia se reemplazo el valor de control lat por su caso extremo");
    control_lat = control_lat_x
#control_notlat
if np.size(data_control_notlat)>0:
    control_notlat = (open_image(data_control_notlat[3][np.argmin(data_control_notlat[0])]),
                      data_control_notlat[2][np.argmin(data_control_notlat[0])],
                      np.min(data_control_notlat[0]))
else:
    print("Advertencia se reemplazo el valor de control notlat por su caso extremo");
    control_notlat = control_notlat_x
    


#LUZ_lat
if np.size(data_LUZ_lat)>0:
    LUZ_lat = (open_image(data_LUZ_lat[3][np.argmax(data_LUZ_lat[0])]),
                   data_LUZ_lat[2][np.argmax(data_LUZ_lat[0])],
                   np.max(data_LUZ_lat[0]))
else:
    print("Advertencia se reemplazo el valor de LUZ lat por su caso extremo");
    LUZ_lat = LUZ_lat_x
#control_notlat
if np.size(data_LUZ_notlat)>0:
    LUZ_notlat = (open_image(data_LUZ_notlat[3][np.argmin(data_LUZ_notlat[0])]),
                      data_LUZ_notlat[2][np.argmin(data_LUZ_notlat[0])],
                      np.min(data_LUZ_notlat[0]))
else:
    print("Advertencia se reemplazo el valor de control notlat por su caso extremo");
    LUZ_notlat = LUZ_notlat_x
    
    
    
    
    
#MEL_lat
if np.size(data_MEL_lat)>0:
    MEL_lat = (open_image(data_MEL_lat[3][np.argmax(data_MEL_lat[0])]),
                   data_MEL_lat[2][np.argmax(data_MEL_lat[0])],
                   np.max(data_MEL_lat[0]))
else:
    print("Advertencia se reemplazo el valor de MEL lat por su caso extremo");
    MEL_lat = MEL_lat_x
#control_notlat
if np.size(data_MEL_notlat)>0:
    MEL_notlat = (open_image(data_MEL_notlat[3][np.argmin(data_MEL_notlat[0])]),
                      data_MEL_notlat[2][np.argmin(data_MEL_notlat[0])],
                      np.min(data_MEL_notlat[0]))
else:
    print("Advertencia se reemplazo el valor de MEL notlat por su caso extremo");
    MEL_notlat = MEL_notlat_x

    
    
    
#LUZ + MEL lat
if np.size(data_MELplusLUZ_lat)>0:
    MELplusLUZ_lat = (open_image(data_MELplusLUZ_lat[3][np.argmax(data_MELplusLUZ_lat[0])]),
                   data_MELplusLUZ_lat[2][np.argmax(data_MELplusLUZ_lat[0])],
                   np.max(data_MELplusLUZ_lat[0]))
else:
    print("Advertencia se reemplazo el valor de MELplusLUZ lat por su caso extremo");
    MELplusLUZ_lat = MELplusLUZ_lat_x
#control_notlat
if np.size(data_MELplusLUZ_notlat)>0:
    MELplusLUZ_notlat = (open_image(data_MELplusLUZ_notlat[3][np.argmin(data_MELplusLUZ_notlat[0])]),
                      data_MELplusLUZ_notlat[2][np.argmin(data_MELplusLUZ_notlat[0])],
                      np.min(data_MELplusLUZ_notlat[0]))
else:
    print("Advertencia se reemplazo el valor de MELplusLUZ notlat por su caso extremo");
    MELplusLUZ_notlat = MELplusLUZ_notlat_x

    
    
    
fig,axs = plt.subplots(4,2)

for ax in axs.flat:
    ax.set(xticks=[], yticks=[])
  
    
fig.set_size_inches(30,60)
fig.suptitle("\n        <-Lateralizado | No Lateralizado ->\n(Cercania al Umbral [%s ; %s] )"%(lat,no_lat),fontsize=55)

fontsize_titles = 50
fontsize_ylabels = 50
fontsize_xlabels = 50

axs[0,0].imshow(control_lat[0][:,:,::-1])
axs[0,0].set_title("\n%s(%s%%)\n"%(control_lat[1],control_lat[2]),fontsize = fontsize_titles)
axs[0,0].set_ylabel(ylabel='Control',fontsize=fontsize_ylabels)


axs[0,1].imshow(control_notlat[0][:,:,::-1])
axs[0,1].set_title("\n%s(%s%%)\n"%(control_notlat[1],control_notlat[2]),fontsize = fontsize_titles)


axs[1,0].imshow(MEL_lat[0][:,:,::-1])
axs[1,0].set_title("\n%s(%s%%)\n"%(MEL_lat[1],MEL_lat[2]),fontsize = fontsize_titles)
axs[1,0].set_ylabel(ylabel='MEL',fontsize=fontsize_ylabels)

axs[1,1].imshow(MEL_notlat[0][:,:,::-1])
axs[1,1].set_title("\n%s(%s%%)\n"%(MEL_notlat[1],MEL_notlat[2]),fontsize = fontsize_titles)



axs[2,0].imshow(LUZ_lat[0][:,:,::-1])
axs[2,0].set_title("\n%s(%s%%)\n"%(LUZ_lat[1],LUZ_lat[2]),fontsize = fontsize_titles)
axs[2,0].set_ylabel(ylabel='LUZ',fontsize=fontsize_ylabels)

axs[2,1].imshow(LUZ_notlat[0][:,:,::-1])
axs[2,1].set_title("\n%s(%s%%)\n"%(LUZ_notlat[1],LUZ_notlat[2]),fontsize = fontsize_titles)



axs[3,0].imshow(MELplusLUZ_lat[0][:,:,::-1])
axs[3,0].set_title("\n%s(%s%%)\n"%(MELplusLUZ_lat[1],MELplusLUZ_lat[2]),fontsize = fontsize_titles)
axs[3,0].set_ylabel(ylabel='MEL + LUZ',fontsize=fontsize_ylabels)

axs[3,1].imshow(MELplusLUZ_notlat[0][:,:,::-1])
axs[3,1].set_title("\n%s(%s%%)\n"%(MELplusLUZ_notlat[1],MELplusLUZ_notlat[2]),fontsize = fontsize_titles)

plt.show();



fig,axs = plt.subplots(4,2)

for ax in axs.flat:
    ax.set(xticks=[], yticks=[])
  
    
fig.set_size_inches(30,60)
fig.suptitle("\n\n        <-Lateralizado | No Lateralizado ->\n(Casos extremos)",fontsize=55)

fontsize_titles = 50
fontsize_ylabels = 50
fontsize_xlabels = 50

axs[0,0].imshow(control_lat_x[0][:,:,::-1])
axs[0,0].set_title("\n%s(%s%%)\n"%(control_lat_x[1],control_lat_x[2]),fontsize = fontsize_titles)
axs[0,0].set_ylabel(ylabel='Control',fontsize=fontsize_ylabels)


axs[0,1].imshow(control_notlat_x[0][:,:,::-1])
axs[0,1].set_title("\n%s(%s%%)\n"%(control_notlat_x[1],control_notlat_x[2]),fontsize = fontsize_titles)


axs[1,0].imshow(MEL_lat_x[0][:,:,::-1])
axs[1,0].set_title("\n%s(%s%%)\n"%(MEL_lat_x[1],MEL_lat_x[2]),fontsize = fontsize_titles)
axs[1,0].set_ylabel(ylabel='MEL',fontsize=fontsize_ylabels)

axs[1,1].imshow(MEL_notlat_x[0][:,:,::-1])
axs[1,1].set_title("\n%s(%s%%)\n"%(MEL_notlat_x[1],MEL_notlat_x[2]),fontsize = fontsize_titles)



axs[2,0].imshow(LUZ_lat_x[0][:,:,::-1])
axs[2,0].set_title("\n%s(%s%%)\n"%(LUZ_lat_x[1],LUZ_lat_x[2]),fontsize = fontsize_titles)
axs[2,0].set_ylabel(ylabel='LUZ',fontsize=fontsize_ylabels)

axs[2,1].imshow(LUZ_notlat_x[0][:,:,::-1])
axs[2,1].set_title("\n%s(%s%%)\n"%(LUZ_notlat_x[1],LUZ_notlat_x[2]),fontsize = fontsize_titles)



axs[3,0].imshow(MELplusLUZ_lat_x[0][:,:,::-1])
axs[3,0].set_title("\n%s(%s%%)\n"%(MELplusLUZ_lat_x[1],MELplusLUZ_lat_x[2]),fontsize = fontsize_titles)
axs[3,0].set_ylabel(ylabel='MEL + LUZ',fontsize=fontsize_ylabels)

axs[3,1].imshow(MELplusLUZ_notlat_x[0][:,:,::-1])
axs[3,1].set_title("\n%s(%s%%)\n"%(MELplusLUZ_notlat_x[1],MELplusLUZ_notlat_x[2]),fontsize = fontsize_titles)

plt.show();

