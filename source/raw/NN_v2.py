#Importaciones
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.utils import plot_model   #Para mostrar grafico del modelo

#carga de datos
df = pd.read_excel("Datos_normalizados.xlsx")

#separar en entrenamiento, validacion y test
spin = 4        #muestras de entrada
prout = 4       #numero de predicciones del modelo
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

#Data de entrenamiento
x_train0 = np.array(train_df[0:len(train_df)-prout],dtype=float)
x_train = np.random.rand(len(train_df)-spin-prout+1,spin,9)
for i in range(0,len(train_df)-spin-prout-1):
    for j in range(0,spin-1):
        for k in range(0,8):
            x_train[i,j,k] = x_train0[i+j,k]
y_train0 = np.array(train_df[spin:len(train_df)].drop(['Hora','Media Red','Media Green','Media Blue'], axis=1),dtype=float)
y_train = np.random.rand(len(train_df)-spin-prout+1,prout,5)
for i in range(spin, len(train_df)-spin-prout-1):
    for j in range(0,prout-1):
        for k in range(0,4):
            y_train[i-spin,j,k] = y_train0[i+j,k]

#Datos de validación
x_val0 = np.array(val_df[0:len(val_df)-prout],dtype=float)
x_val = np.random.rand(len(val_df)-spin-prout+1,spin,9)
for i in range(0,len(val_df)-spin-prout-1):
    for j in range(0,spin-1):
        for k in range(0,8):
            x_val[i,j,k] = x_val0[i+j,k]
y_val0 = np.array(val_df[spin:len(val_df)].drop(['Hora','Media Red','Media Green','Media Blue'], axis=1),dtype=float)
y_val = np.random.rand(len(val_df)-spin-prout+1,prout,5)
for i in range(spin, len(val_df)-spin-prout-1):
    for j in range(0,prout-1):
        for k in range(0,4):
            y_val[i-spin,j,k] = y_val0[i+j,k]

#Datos de testeo
x_test0 = np.array(test_df[0:len(test_df)-prout],dtype=float)
x_test = np.random.rand(len(test_df)-spin-prout+1,spin,9)
for i in range(0,len(test_df)-spin-prout-1):
    for j in range(0,spin-1):
        for k in range(0,8):
            x_test[i,j,k] = x_test0[i+j,k]
y_test0 = np.array(test_df[spin:len(test_df)].drop(['Hora','Media Red','Media Green','Media Blue'], axis=1),dtype=float)
y_test = np.random.rand(len(test_df)-spin-prout+1,prout,5)
for i in range(spin, len(test_df)-spin-prout-1):
    for j in range(0,prout-1):
        for k in range(0,4):
            y_test[i-spin,j,k] = y_test0[i+j,k]

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

#Crear Modelo y Guardarlo
# #Modelo de prueba
# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[4,9]))      #capa de entrada
# model.add(keras.layers.Dense(62, activation="relu"))    #capa oculta 1
# model.add(keras.layers.Dense(36, activation="relu"))    #capa oculta 2
# model.add(keras.layers.Dense(18, activation="relu"))    #capa oculta 3
# model.add(keras.layers.Dense(5, activation="softmax"))  #capa de salida
# model.save("Modelo_Prueba.h5")                          #guardo modelo
#Modelo LSTM
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=200, activation = 'relu', input_shape=[spin,9]))
model.add(keras.layers.Dense(5*prout))
model.add(keras.layers.Reshape((prout,5)))

model.save("Modelo_Prueba.h5")

#Cargar modelo
model = keras.models.load_model("Modelo_Prueba.h5")

#Muestra en Consola
model.summary()     #Muestra las capas del modelo, con todos sus parametros
tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)     #muestra grafico del modelo

#Seleccion de la funcion de perdida y Optimizador
#model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='binary_crossentropy')
model.compile(optimizer=keras.optimizers.Adam(0.001),loss='mse')

#Entrenamiento
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

# #Graficar Perdida y Precision durante el proceso de entrenamiento
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1) # Rango Vertical [0-1]

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(history.history["loss"])

# #Evaluar el error del modelo (Perdida y Precision)
# model.evaluate(x_test, y_test)

#Probar Modelo con datos de verificacion
X_new = np.reshape(x_test[10], [1,spin,9])
y_proba = model.predict(X_new)
print(y_proba)
print(y_test[0])

plt.show()