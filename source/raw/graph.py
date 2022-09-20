import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import datetime as dt
import os
from image_pr import *

plt.style.use(['dark_background'])      #configuración de la ventana de los plots

#leer los datos obtenidos
dir = os.getcwd()
datos = pn.read_excel(f'./Datos/Datos_Meteorologicos.xlsx')
ind = datos.shape[0]                    #cantidad de datos totales

#establecer los valores minimos y maximos de las variables
norm_limit = {'t_min':-10, 't_max':45, 'p_min':850, 'p_max':1200, 'h_min':0, 'h_max':100, 'ws_min':0, 'ws_max':200, 'wd_min':0, 'wd_max':360}

#configurar e inicializar variables
temp = np.array(datos['Temperatura'])
pres = np.array(datos['Presion'])
hum = np.array(datos['Humedad'])
wind_speed = np.array(datos['Velocidad del Viento'])
wind_dir = np.array(datos['Direccion del Viento'])
dates = np.array(datos['Fecha'])
hora = np.array(datos['Hora'])
temp_n = []
pres_n = []
hum_n = []
ws_n = []
wd_n = []
hora_n = []
datex = []
fecha = []
a = 1
locx = [0]
datos_norm = []

#bucle for para normalizar y preparar variables para graficar
for i in range(0,ind):
    #normalización de variables meteorologicas
    temp_n.append((temp[i]-norm_limit['t_min'])/(norm_limit['t_max']-norm_limit['t_min']))
    pres_n.append((pres[i]-norm_limit['p_min'])/(norm_limit['p_max']-norm_limit['p_min']))
    hum_n.append((hum[i]-norm_limit['h_min'])/(norm_limit['h_max']-norm_limit['h_min']))
    ws_n.append((wind_speed[i]-norm_limit['ws_min'])/(norm_limit['ws_max']-norm_limit['ws_min']))
    wd_n.append((wind_dir[i]-norm_limit['wd_min'])/(norm_limit['wd_max']-norm_limit['wd_min']))

    #normalización de la hora
    date_time = dt.datetime.strptime(hora[i], "%H:%M:%S")
    delta = date_time - dt.datetime(1900, 1, 1)
    seconds = delta.total_seconds()
    hora_n.append(seconds/86399)

    #procesamiento de imagen en RGB
    name = f'./Datos/Fotos Cam/{dates[i]}_{hora[i]}.png'
    name = name[slice(None, None, -1)].replace(':', '', 2)
    mp = immedRGB(name[slice(None, None, -1)])

    #matriz con los datos normalizados
    datos_norm.append([temp_n[i], pres_n[i], hum_n[i], ws_n[i], wd_n[i], hora_n[i], mp[0], mp[1], mp[2]])

    #preparación de base de tiempo para graficar
    fecha.append(f'{dates[i]} {hora[i]}')
    if i != 0:
        if datex[a-1] != dates[i]:
            datex.append(dates[i])
            locx.append(i)
            a = a+1
    else:
        datex.append(dates[0])
    
    print(f'{i} de {ind-1}')

xlen = len(datex)               #división de la base de tiempo para graficar

#guardado de datos normalizados en formato .csv y .xlsx
datos_norm = pn.DataFrame(data=datos_norm, columns=['Temperatura', 'Presión', 'Humedad', 'Velocidad del Viento', 'Dirección del Viento', 'Hora', 'Media Red', 'Media Green', 'Media Blue'])
datos_norm.to_csv('Datos_normalizados.csv', index=False)
datos_norm.to_excel('Datos_normalizados.xlsx', index=False)

#graficos de cada variable meteorologica
plt.subplot(231)
plt.plot(fecha, temp, 'r-')
plt.ylabel('Temperatura')
locs, labels = plt.xticks()
plt.xticks(locx, datex, rotation=60)
plt.grid(visible=True, linestyle='-.')

plt.subplot(232)
plt.plot(fecha, pres, 'b-')
plt.ylabel('Presión')
locs, labels = plt.xticks()
plt.xticks(locx, datex, rotation=60)
plt.grid(visible=True, linestyle='-.')

plt.subplot(233)
plt.plot(fecha, hum, 'c-')
plt.ylabel('Humedad')
locs, labels = plt.xticks()
plt.xticks(locx, datex, rotation=60)
plt.grid(visible=True, linestyle='-.')

plt.subplot(234)
plt.plot(fecha, wind_speed, 'g-')
plt.ylabel('Velocidad del viento')
locs, labels = plt.xticks()
plt.xticks(locx, datex, rotation=60)
plt.grid(visible=True, linestyle='-.')

plt.subplot(235)
plt.plot(fecha, wind_dir, 'y-')
plt.ylabel('Dirección del viento')
locs, labels = plt.xticks()
plt.xticks(locx, datex, rotation=60)
plt.grid(visible=True, linestyle='-.')

plt.show()
