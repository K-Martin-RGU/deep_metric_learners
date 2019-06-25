import numpy as np
import math
import random
from operator import itemgetter
import heapq
from sklearn.metrics.pairwise import cosine_similarity

#For reproducability
np.random.seed(1337)

#Keras imports
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import SelfbackUtil as inn
import read as PAMAP
    
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

#Define Euclidean distance function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis = 1, keepdims = True))
    
#Define the shape of the output of Euclidean distance
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
        
#Define the contrastive loss function (as from Hadsell et al [1].)
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices, num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
    
def create_random_pairs(x, y):
    pairs = []
    labels = []
    
    for i in range(len(x)):
        rand_pos_pair = x[random.choice(np.flatnonzero(y == y[i]))]
        rand_neg_pair = x[random.choice(np.flatnonzero(y != y[i]))]
        pairs += [[x[i], rand_pos_pair]]
        pairs += [[x[i], rand_neg_pair]]
        labels += [1, 0]
        
    return np.array(pairs), np.array(labels)

def build_mlp_model(input_shape):

    base_input = Input((input_shape,))
    x = Dense(1200, activation='relu')(base_input)
    embedding_model = Model(base_input, x, name='embedding')
    
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

#Because we use the same instance 'base network' the weights of the network will 
#be shared across the two branches
    processed_a = embedding_model(input_a)
    processed_b = embedding_model(input_b)

    distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([processed_a, processed_b])

    siamese_model = Model(input = [input_a, input_b], output = distance)

#Train (Compile and Fit) the model
    #rms = RMSprop(lr = 0.0001)
    siamese_model.compile(loss = contrastive_loss, optimizer = 'adam')

 
    return embedding_model, siamese_model
    
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
writepath = "Results/PAMAP_Siamese_Results.txt"
loov_scores = []

for person in data.keys():
    
    x_train, y_train, x_test, y_test = loov(data, person)

    input_dim = 540
    nb_epoch = 1
    num_classes = 8

#Network definition
    base_network, siamese_model = build_mlp_model(input_dim)

    for x in range(20):
        #Create training and test positive and negative pairs
        digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
        x_pairs, y_pairs = create_pairs(x_train, digit_indices, num_classes)
    
        siamese_model.fit([x_pairs[:, 0], x_pairs[:, 1]], y_pairs, 
                          batch_size = 128, epochs = nb_epoch)
    
    #Get the new representation of data from the layer before similarity
    x_pred = base_network.predict(x_train)
    x_t = base_network.predict(x_test)
    
    
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