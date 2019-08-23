import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import read
import sys
from keras.utils import np_utils

import heapq
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(1)
tf.set_random_seed(2)

feature_length = read.dct_length * 3 * 3
batch_size = 60
epochs = 10

def cos_knn(k, test_data, test_labels, stored_data, stored_target):
    cosim = cosine_similarity(test_data, stored_data)

    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    top = [[stored_target[j] for j in i[:k]] for i in top]

    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    return correct / float(len(test_labels))

def split(_data, _test_ids):
    train_data_ = {key: value for key, value in _data.items() if key not in _test_ids}
    test_data_ = {key: value for key, value in _data.items() if key in _test_ids}
    return train_data_, test_data_


def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return flatten_data, flatten_labels


def mlp():
    _input = Input(shape=(feature_length,))
    x = Dense(1200, activation='relu')(_input)
    x = BatchNormalization()(x)
    embedding = Model(inputs=_input, outputs=x, name='embedding')
    
    x = Dense(len(read.classes), activation='softmax')(x)
    classifier = Model(inputs=_input, outputs=x, name='classifier')
    
    return embedding, classifier

feature_data = read.read()
test_ids = list(feature_data.keys())

for test_id in test_ids:

    _train_data, _test_data = split(feature_data, test_id)
    _train_data, _train_labels = flatten(_train_data)
    _test_data, _test_labels = flatten(_test_data)
    
    _train_data = np.array(_train_data)
    _test_data = np.array(_test_data)
    
    _test_labels_ = np_utils.to_categorical(_test_labels, len(read.classes))
    _train_labels_ = np_utils.to_categorical(_train_labels, len(read.classes))
    
    embedding, _model = mlp()

    _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    _model.fit(_train_data, _train_labels_, verbose=1, batch_size=batch_size, epochs=epochs, shuffle=True)

    _train_preds = embedding.predict(_train_data)
    _test_preds = embedding.predict(_test_data)    

    # classifier evaluation
    results = _model.evaluate(_test_data, _test_labels_, batch_size=batch_size, verbose=0)
    print(results)

    # knn evaluation
    k = 3
    acc = cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)
    print(acc)
    
    read.write_data('mlp.csv', 'score'+','+','.join([str(f) for f in results])+','+'knn_acc'+','+str(acc))