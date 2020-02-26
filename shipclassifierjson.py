# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 00:41:40 2018

@author: Amartya
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
#%%
#source = 'C:\\Users\\Amartya\\TUM\\99 Projects\\02 Kaggle ships\\Kaggle Data\\'
#dest = 'C:\\Users\\Amartya\\TUM\\99 Projects\\02 Kaggle ships\\Record Data\\'
file = open(source + 'shipsnet.json')
dataset = json.load(file)
file.close()
print('File read')
#%%
image_data = np.array(dataset['data']).astype('uint8')
label_data = np.array(dataset['labels']).astype('uint8')
#%%
NUM_FILES = image_data.shape[0]
NUM_CLASSES = 2
#%%
indices = np.random.permutation(NUM_FILES)
train_split = (NUM_FILES * 80) // 100
train_idx, test_idx = indices[:train_split], indices[train_split:]

train_images = np.take(image_data, train_idx,0)
test_images = np.take(image_data, test_idx,0)

train_labels = np.take(label_data, train_idx)
test_labels = np.take(label_data, test_idx)
#%%
train_images = train_images.reshape([-1,3,80,80]).transpose([0,2,3,1])
test_images = test_images.reshape([-1,3,80,80]).transpose([0,2,3,1])

#%%
model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=2,padding='same',activation=tf.nn.relu,input_shape=(80,80,3)),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(filters=32, kernel_size=2,padding='same',activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
])
#%%
model.summary()
#%%
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
#%%
model.fit(train_images, train_labels, batch_size=64, epochs=5)
#%%
test_images = np.expand_dims(test_images, axis=3)
#%%
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)