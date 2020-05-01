import plot_keypoints as kplot
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import re

Participant = "Fotis"
Trials_num = 3

'''
Labels:
    1: cube
    2: polygon
    3: cup
    4: bottle
    5: plate
'''
#
# gestures = ['cube', 'polygon', 'cup', 'bottle', 'plate']
# gesture_dict = {'cube': 1, 'polygon': 2, 'cup': 3, 'bottle': 4, 'plate': 5}
# #
# gestures = ['cube', 'polygon', 'cup']
# gesture_dict = {'cube': 1, 'polygon': 2, 'cup': 3}
#
gestures = ['cube', 'polygon']
gesture_dict = {'cube': 1, 'polygon': 2}


def load_files_heracleia(max_time_steps=130):
    x_data = None
    labels = []
    for i in range(Trials_num):
        for gesture in gestures:
            sample = kplot.new_keypointExtractor(
                "~/Experiment2_csv/kp_" + gesture + "_" + Participant + str(i + 1) + ".csv").values[10:max_time_steps]
            if x_data is None:
                x_data = np.reshape(sample, (-1, sample.shape[0], sample.shape[1]))
            else:
                x_data = np.append(x_data, np.reshape(sample, (-1, sample.shape[0], sample.shape[1])), axis=0)
            labels.append(gesture_dict.get(gesture))

    # print(x_data.shape)
    return x_data, np.asarray(labels)


def extractFeatures(columns):
    col = columns[1:].values
    cur = None
    features = []
    for name in col:
        # striped_name = name.split("_")[:-2]
        striped_name = re.findall(r'(\w+?)(\d+)', name)[0][0][:-1]

        if cur != striped_name:
            features.append(striped_name)
            cur = striped_name

    return features


def load_files_object_size():
    x_data = None
    labels = []

    file = "/mnt/34C28480C28447D6/Datasets/KinematicsData.xlsx"
    df = pd.read_excel(file)

    # reformat data to be in shape (#samples, time_steps, features) = (848,10,12)
    features = extractFeatures(df.columns)
    for feature in features:
        relative_columns = df.columns.to_series().str.contains(feature)
        var = df[df.columns[relative_columns]].values
        if x_data is None:
            x_data = np.reshape(var, (var.shape[0], var.shape[1], -1))
        else:
            x_data = np.append(x_data, np.reshape(var, (var.shape[0], var.shape[1], -1)), axis=2)

    labels = df['Object Size'].values
    # normalize labels
    labels = [label-1 for label in labels]

    # print(x_data.shape)
    return x_data, np.asarray(labels)


def prepare_dataset(X, Y):
    kf = KFold(n_splits=5, shuffle=False)  # Define the split - into 2 folds
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        labels_train, labels_test = Y[train_index], Y[test_index]
    return X_train, labels_train, X_test, labels_test


#x_data, labels = load_files_object_size()
# X_train, labels_train, X_test, labels_test = prepare_dataset(x_data, labels)
#print()
