# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The image dataset we taken is mnist and the model must denoise the images and show it with better quality and remove the unwanted noises and learn to show the better version of the images.

Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.The goal of an autoencoder is to get an output that is identical to the input. MNIST is a dataset of black and white handwritten images of size 28x28.Denoising is the process of removing noise. This can be an image, audio, or document.These noisy digits will serve as our input data to our encoder. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image.

![201460551-99b57c03-8dd9-4ec2-9d7c-1772acf980c4](https://github.com/Dhanudhanaraj/convolutional-denoising-autoencoder/assets/119218812/a76c00d7-cbd5-4527-a5c0-571758e45f02)

## DESIGN STEPS

### STEP 1:
Download and split the dataset into training and testing datasets.
### STEP 2:
Rescale the data so that the training is made easier.
### STEP 3:
We create two networks , one for encoding and one for decoding.


## PROGRAM
```
NAME:DHANUMALYA.D
REGISTER NUMBER:212222230030
```
### Import required libraries
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```
### Loading, preprocessing and adding some noise to the data
```
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
 ```    
### Creating model, compile and fitting it
```
input_img = keras.Input(shape=(28, 28, 1))
conv1  = layers.Conv2D(32,(3,3),activation ='relu',padding = 'same')(input_img)
maxpool1 = layers.MaxPooling2D((2,2),padding='same')(conv1)
conv2  = layers.Conv2D(32,(3,3),activation ='relu',padding = 'same')(maxpool1)
maxpool2 = layers.MaxPooling2D((2,2),padding='same')(conv1)
conv3  = layers.Conv2D(32,(3,3),activation ='relu',padding = 'same')(maxpool2)
maxpool3 = layers.MaxPooling2D((2,2),padding='same')(conv3)
conv4  = layers.Conv2D(32,(3,3),activation ='relu',padding = 'same')(maxpool3)
upsamp1 = layers.UpSampling2D((2,2))(conv4)
conv5  = layers.Conv2D(32,(3,3),activation ='relu',padding = 'same')(upsamp1)
upsamp2 = layers.UpSampling2D((2,2))(conv5)
output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsamp2)

autoencoder = keras.Model(input_img, output)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```
### Predict the noisy added test data
```
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
## OUTPUT

### Original vs Noisy Vs Reconstructed Image
![Screenshot 2024-05-05 232829](https://github.com/Dhanudhanaraj/convolutional-denoising-autoencoder/assets/119218812/3893326e-dee5-42a6-bfc4-5fbfa3d3bd40)


## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
