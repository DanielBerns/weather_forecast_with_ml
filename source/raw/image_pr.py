import numpy as np
from PIL import Image

#Función para procesar la imagen y calcular la media aritmetica de cada matriz 
def immedRGB(name):
    img1 = Image.open(name)                     #abro imagen
    imnp1n = np.asarray(img1)                   #convierto la imagen a numpy array
    imnp1n = imnp1n/255                         #normalizo, salida entre 0.0 y 1.0
    dim = imnp1n.shape                          #(480, 640, 3)
    at = dim[0]*dim[1]                          #at = cantidad total de elementos por matriz

    #inicializo variables
    sumR = 0
    sumG = 0
    sumB = 0

    #bucle for for para calcular sumatoria de todos los elementos de cada matriz
    for i in range(dim[0]):
        for j in range(dim[1]):
            sumR = imnp1n[i,j,0] + sumR
            sumG = imnp1n[i,j,1] + sumG
            sumB = imnp1n[i,j,2] + sumB

    #calculo media aritmetica de cada matriz
    sumR = sumR/at
    sumG = sumG/at
    sumB = sumB/at

    return [sumR, sumB, sumG]

#Fourier transform breaks down an image into sine and cosine components. 
#   It has multiple applications like image reconstruction, image compression, or image filtering. 
#   Since we are talking about images, we will take discrete fourier transform into consideration.

#Image gradients can be used to extract information from images. Gradient images are created from the original 
# image (generally by convolving with a filter, one of the simplest being the Sobel filter) for this purpose. 
# Each pixel of a gradient image measures the change in intensity of that same point in the original image, in 
# a given direction. To get the full range of direction, gradient images in the x and y directions are 
# computed.