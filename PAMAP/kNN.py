import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

    return pred

def get_neighbours(instance, dataset, n):
    return np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]

def get_accuracy(test_labels, predictions):
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == predictions[j]:
            correct += 1
    return (correct/float(len(test_labels))) * 100.0


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
writepath = "Results/PAMAP_kNN_Results.txt"
loov_scores = []

for person in data.keys():
    
    x_train, y_train, x_test, y_test = loov(data, person)
    predictions = []
    k = 3
            
    predictions = cos_knn(k, x_test, y_test, x_train, y_train)
                        
    print(len(predictions))
    acc = get_accuracy(y_test, predictions)
            
    inn.write_data(writepath, str(acc))
    print("Accuracy: ", acc)
    loov_scores += [acc]

print(loov_scores)
loov_scores = np.array(loov_scores)
print("Average accuracy: " + str(np.mean(loov_scores)))
