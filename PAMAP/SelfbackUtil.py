import os
import numpy as np
import random
import csv

np.random.seed(1234)

activityType = ["jogging", "sitting", "standing", "walking", "upstairs", "downstairs"]
idList = range(len(activityType))
activityIdDict = dict(zip(activityType, idList))

zeros = [[0]*3]*300
ones = [[1]*3]*300

def extract_features(train_data, is_test=False, samp_rate=100, window_length=3):
    wrist_windows = []
    labels = []
    for data in train_data:
        for activity in data:
            df = data[activity]
            w_windows = split_windows(df, samp_rate, window_length, is_test=is_test)
            wrist_windows.extend(w_windows)
            for i in range(len(w_windows)):
                labels.append(activityIdDict.get(activity))
    y = labels
    return wrist_windows, y

def split_windows(data, samp_rate, window_length, is_test=False):
    wrist_windows = []
    width = samp_rate * window_length  # 100 * 3
    i = 0
    N = len(data)
    while i + width < N:
        start = i
        end = start + width
        w = [a[:3] for a in data[start:end]]
        i = int(i + (width))
        wrist_windows.append(w)
    return wrist_windows

def read_data(path):
    person_data = {}

    files = os.listdir(path)
    for f in files:
        temp = f.split("_")
        user = temp[0]
        activity = temp[1]
        data = []
        reader = csv.reader(open(os.path.join(path, f), "r"), delimiter=",")
        for row in reader:
            data.append(row)

        activity_data = {}
        if user in person_data:
            activity_data = person_data[user]
            activity_data[activity] = data
        else:
            activity_data[activity] = data
        person_data[user] = activity_data

    return person_data

def write_data(file_path, data):
    if(os.path.isfile(file_path)):
        f = open(file_path, 'a')
        f.write(data+'\n')
    else:
        f = open(file_path, 'w')
        f.write(data+'\n')
    f.close()

def get_keeparray(keep_prob, size):
    array = [False] * size
    drop_set = set()
    keep_size = int( keep_prob*size)
    while len(drop_set) < keep_size:
        drop_set.add(np.random.randint(0,size))
    for el in drop_set:
        array[el] = True
    return array

def update_thigh_shape(thigh, keep_prob):
    new_thigh = []
    array = get_keeparray(keep_prob, len(thigh))
    for keep,instance in zip(array, thigh):
        if keep:
            new_thigh.append(instance)
        else:
            new_thigh.append(zeros)
    return new_thigh

def create_zeros(array):
    a = [[[0]*3 for el in ele] for ele in array]
    return a

def update_by_diff(thigh, index_diff, keep_prob):
    new_thigh = []
    index_diff = sorted(index_diff.items(), key=lambda x: x[1])
    keep_prob = int(keep_prob*len(thigh))
    index_diff = [x[0] for x in index_diff[:keep_prob]]
    index = 0
    for t in thigh:
        if index in index_diff:
            new_thigh.append(t)
        else:
            new_thigh.append(zeros)
        index+=1
    return new_thigh

def update_by_diff_order(w_train, t_train, y_train, index_diff, keep_prob):
    wrist_dict = {}
    thigh_dict = {}
    y_dict = {}
    for n, w, t, y in zip(range(len(w_train)), w_train, t_train, y_train):
        wrist_dict[n] = w
        thigh_dict[n] = t
        y_dict[n] = y

    array = get_keeparray(keep_prob, len(thigh_dict))
    for count,keep in zip(range(len(array)), array):
        if not keep:
            thigh_dict[count] = zeros

    index_diff = sorted(index_diff.items(), key=lambda x: x[1])
    print(index_diff.keys())

    ww_train = []
    tt_train = []
    yy_train = []
    for n in index_diff:
        ww_train.append(wrist_dict[n[0]])
        tt_train.append(thigh_dict[n[0]])
        yy_train.append(y_dict[n[0]])
    return ww_train, tt_train, yy_train


def readFile(fileFullPath):
    list = []
    with open(fileFullPath, 'r') as csvfile:
        i = True
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if i:
                i = False
                continue
            list.append(row[:4])
    return list

def read_raw_data(path):
    data = {}
    for f in os.listdir(path):
        newf = os.path.join(path, f)
        if os.path.isdir(newf):
            users = {}
            for ff in os.listdir(newf):
                if ff == '.DS_Store':
                    continue
                name = ff.replace(".csv", "")
                newff = os.path.join(newf, ff)
                users[name] = newff
            data[f] = users
    new_data = {}
    for temp in data:
        activity = temp
        users = data[activity]
        for u in users:
            user = u
            tempDataFile = users[user]
            wristCSV = readFile(tempDataFile)
            if user in new_data:
                activities = new_data[user]
                activities[activity] =  [a[1:4] for a in wristCSV]
                new_data[user] = activities
            else:
                activities = {}
                activities[activity] = [a[1:4] for a in wristCSV]
                new_data[user] = activities
    return new_data


