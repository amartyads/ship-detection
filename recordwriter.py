# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 21:34:31 2018

@author: Amartya
"""

import tensorflow as tf
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
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#%%
indices = np.random.permutation(NUM_FILES)
train_split = (NUM_FILES * 80) // 100
train_idx, test_idx = indices[:train_split], indices[train_split:]

train_images = np.take(image_data, train_idx,0)
test_images = np.take(image_data, test_idx,0)

train_labels = np.take(label_data, train_idx)
test_labels = np.take(label_data, test_idx)
#%%
recordname = 'kaggleshiptrain.tfrecord'
with tf.python_io.TFRecordWriter(recordname) as writer:
    for i in range(0, len(train_images)):
        feature = { 'label': _int64_feature(train_labels[i]),
                           'image': _bytes_feature(train_images[i].tostring())}
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        
        writer.write(example.SerializeToString())
print('Written ' + recordname)
#%%
recordname = 'kaggleshiptest.tfrecord'
with tf.python_io.TFRecordWriter(recordname) as writer:
    for i in range(0, len(test_images)):
        feature = { 'label': _int64_feature(test_labels[i]),
                           'image': _bytes_feature(test_images[i].tostring())}
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        
        writer.write(example.SerializeToString())
print('Written ' + recordname)