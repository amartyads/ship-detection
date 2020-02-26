# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 22:35:52 2018

@author: Amartya
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#source = 'C:\\Users\\Amartya\\TUM\\99 Projects\\02 Kaggle ships\\Record Data\\'
NUM_FILES = 4000
NUM_CLASSES = 2
BATCH_SIZE = 16
#%%
data_feature = { 'image': tf.FixedLenFeature([], dtype = tf.string),
        'label': tf.FixedLenFeature([], dtype = tf.int64)}

def preprocess_fn(record):
    features = tf.parse_single_example( record, features= data_feature )
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.cast(features['label'], tf.int64)
    
    image = tf.cast(image, tf.float32)
    x = tf.reshape(image, (3, 80, 80))
    x = tf.transpose(x, perm=[1,2,0])
    x = tf.image.per_image_standardization(x)
    #x = x / tf.cast(255.0, tf.float32)
    y = tf.one_hot(tf.cast(label, tf.uint8), NUM_CLASSES)
    return x, y
#%%
def get_data(datablock, filenames):    
    comb_set = tf.data.TFRecordDataset(filenames)
    
    if (datablock =='train') | (datablock == 'valid'):
        comb_set = comb_set.shuffle(20000)
    
    comb_set = comb_set.apply(tf.contrib.data.map_and_batch(
        preprocess_fn, BATCH_SIZE,
        drop_remainder=False))
    
    
    comb_set = comb_set.repeat()
    comb_set = comb_set.prefetch(tf.contrib.data.AUTOTUNE)
    
    return comb_set
#%%
train_data = get_data('train', source + 'kaggleshiptrain.tfrecord')
test_data = get_data('test', source + 'kaggleshiptest.tfrecord')

#%%
model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=3,padding='same',input_shape=(80,80,3)),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(filters=64, kernel_size=3,padding='same'),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(filters=64, kernel_size=3,padding='same'),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(filters=64, kernel_size=3,padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=[2,2],strides=2),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
])
#%%
model.summary()
#%%
model.compile('adamax', 'categorical_crossentropy', metrics=['acc'])
#%%
model.fit(train_data, steps_per_epoch = 50, epochs=10, verbose = 1)
#%%
test_loss, test_acc = model.evaluate(test_data, steps=50)

print('Test accuracy:', test_acc)
#%%
predictions = model.predict(test_data, steps=100)
#%%
class_names = ['Not ship', 'Ship']
plt.figure(figsize=(10,10))
iterator = test_data.make_initializable_iterator()

next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(25):
        img, label = sess.run(next_element)
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        op_img = img[0]
        #op_img = op_img[:,:,0]
        
        predicted_label = np.argmax(predictions[i*BATCH_SIZE])
        true_label = np.argmax(label[0])
        if predicted_label == true_label:
          color = 'green'
        else:
          color = 'red'
        plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                      class_names[true_label]),
                                      color=color)
        plt.imshow(op_img)