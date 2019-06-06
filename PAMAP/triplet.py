from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, BatchNormalization, Embedding, Conv1D, GlobalMaxPooling1D
from keras import backend as K
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import SelfbackUtil as inn
import read as PAMAP

random.seed = 1337
np.random.seed = 1337

### k-NN Functions ###
def cos_knn(k, test_data, test_target, stored_data, stored_target):
    """k: number of neighbors to use for voting
    test_data: a set of unobserved images to classify
    test_target: the labels for the test_data (for calculating accuracy)
    stored_data: the images already observed and available to the model
    stored_target: labels for stored_data
    """
    
    # find cosine similarity for every point in test_data between every other point in stored_data
    cosim = cosine_similarity(test_data, stored_data)
    
    # get top k indices of images in stored_data that are most similar to any given test_data point
    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    # convert indices to numbers using stored target values
    top = [[stored_target[j] for j in i[:k]] for i in top]
    
    # vote, and return prediction for every image in test_data
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)
    
    # print table giving classifier accuracy using test_target
    return pred

def get_neighbours(instance, dataset, n):
    return np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]

def get_accuracy(test_labels, predictions):
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == predictions[j]:
            correct += 1
    return (correct/float(len(test_labels))) * 100.0

### Triplet Functions ###
def get_triples_minibatch_indices_me(dictionary):
    triples_indices = []
    for k in dictionary.keys():
        for value in dictionary[k]:
            anchor = value
            positive = random.choice(dictionary[k])
            negative_labels = np.arange(8)
            negative_label = random.choice(np.delete(negative_labels, np.argwhere(negative_labels==k)))
            negative = random.choice(dictionary[negative_label])
            triples_indices.append([anchor, positive, negative])
                
    return np.asarray(triples_indices)

def get_triples_minibatch_data_u(x, dictionary):
    indices = get_triples_minibatch_indices_me(dictionary)
    return x[indices[:,0]], x[indices[:,1]], x[indices[:,2]]

def triplet_generator_minibatch(x, y, no_minibatch):
    grouped = defaultdict(list)
    dict_list = []
    
    for i, label in enumerate(y):
        grouped[label].append(i)
        
    for k in range(len(grouped)):
        random.shuffle(grouped[k])
        
    for j in range(no_minibatch):
        dictionary = {}
        
        for k in range(len(grouped)):
            ran_sam = random.sample(grouped[k], 3)
            dictionary[k] = ran_sam
            
        dict_list.append(dictionary)
    
    i = 0
    
    while 1: 
        x_anchor, x_positive, x_negative = get_triples_minibatch_data_u(x, dict_list[i])
        
        if i == (no_minibatch - 1):
            i = 0
        else:
            i += 1

        yield ({'anchor_input': x_anchor,
               'positive_input': x_positive,
               'negative_input': x_negative},
               None)
        
### Triplet Loss ###
def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

### Embedding Functions ###
def build_mlp_model(input_shape):

    base_input = Input(input_shape)
    x = Dense(1200, activation='relu')(base_input)
    embedding_model = Model(base_input, x, name='embedding')
    
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
   
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
 
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
    
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    triplet_model.compile(loss=None, optimizer='adam') #loss should be None
 
    return embedding_model, triplet_model

def build_cnn_model(input_shape):

    base_input = Input(input_shape)
    x = Dense(128, activation='relu')(base_input)
    embedding_model = Model(base_input, x, name='embedding')
    
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
   
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
 
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
    
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    triplet_model.compile(loss=None, optimizer='adam') #loss should be None
 
    return embedding_model, triplet_model

### Dataset Functions ###
def read_full_dataset(indice_keys = None):
    data = PAMAP.read()
    return data

def loov(data, test_person):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    test_data = data[test_person]
    for label in test_data.keys():
        for example in test_data[label]:
            x_test += [example]
            y_test += [label]
    
    copy = data.copy()
    copy.pop(test_person, None)
    
    for people in copy.keys():
        person = copy[people]
        for label in person.keys():
            for example in person[label]:
                x_train += [example]
                y_train += [label]
            
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

### Main code ###

data = read_full_dataset()
writepath = "Results/PAMAP_Triplet_Results.txt"
loov_scores = []

for person in data.keys():
    
    x_train, y_train, x_test, y_test = loov(data, person)
    print(len(x_train))
    print(set(y_train))
    
    no_minibatch = 200
    batch_size = 128
    steps_per_epoch = no_minibatch
     
    embedding_model, triplet_model = build_mlp_model((540,))
    
    history = triplet_model.fit_generator(triplet_generator_minibatch(x_train, y_train, no_minibatch),
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=20,
                                          verbose=1)

    x_pred = embedding_model.predict(x_train)
    x_t = embedding_model.predict(x_test)
    
    predictions = []
    k = 3
            
    predictions = cos_knn(k, x_t, y_test, x_pred, y_train)
                        
    print(len(predictions))
    acc = get_accuracy(y_test, predictions)
            
    inn.write_data(writepath, str(acc))
    print("Accuracy: ", acc)
    loov_scores += [acc]

print(loov_scores)
loov_scores = np.array(loov_scores)
print("Average accuracy: " + str(np.mean(loov_scores)))