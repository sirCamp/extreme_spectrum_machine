from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(path, filename, type='npy'):
    train_data = np.load(path + '/' + filename + 'training_data.'+type)
    test_data = np.load(path + '/' + filename + 'test_data.'+type)

    train_data_label = np.load(path + '/' + filename + 'training_labels.'+type)
    test_data_label = np.load(path + '/' + filename + 'test_labels.'+type)

    return train_data, test_data, train_data_label, test_data_label


def load_csv_data(path, filename, type='csv', sep=",", id=False):
    df = pd.read_csv(path + '/' + filename + 'training_data.'+type, sep=sep, engine='python')
    df['index'] = range(1, len(df) + 1)

    df = df[['index','Sentiment', 'SentimentText']]

    labels = df[['Sentiment']].values.reshape(-1)
    if id:
        data = df[['index','SentimentText']].values  # .reshape(-1)
    else:
        data = df[['SentimentText']].values.reshape(-1)


    return data, labels



def load_csv_data_filter(path, filename, type='csv', sep=",", id=False):
    df = pd.read_csv(path + '/' + filename +'training_data.'+type, sep=sep, engine='python',)

    df = df[['id', 'target', 'text']]

    labels = df[['target']].values.reshape(-1)
    if id:
        data = df[['id','text']].values
    else:
        data = df[['text']].values.reshape(-1)

    stratified_sampling(data, labels)


    return data, labels


def stratified_sampling(X, y, output_name):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.0075, random_state=42)

    dictionary = {}
    for dataset_index, other_index in sss.split(X, y):

            ssss = StratifiedShuffleSplit(n_splits=1, train_size=5/float(6), test_size=1/float(6), random_state=1234)
            X_dataset = X[dataset_index]
            y_dataset = y[dataset_index]
            for train_train_index, test_index in ssss.split(X_dataset, y_dataset):

                sssss = StratifiedShuffleSplit(n_splits=1, test_size=1/float(10), train_size=9/float(10), random_state=1234)
                X_train = X_dataset[train_train_index]
                y_train = y_dataset[train_train_index]
                for train_index, val_index in sssss.split(X_train, y_train):
                    dictionary["dataset_index"] = dataset_index
                    dictionary["train_train_index"] = train_train_index
                    dictionary["train_index"] = train_index
                    dictionary["test_index"] = test_index
                    dictionary["val_index"] = val_index
                    break
                break
            break

    np.save(output_name,dictionary)


def data_encoding_transformation(func, data, array=True):
    data_transformed = []

    for j in range(len(data)):
        row = data[j]
        data_transformed.append(func(row, array))

    return data_transformed


def to_categorical_values(func, data):
    data_transformed = []

    for j in range(len(data)):
        row = data[j]
        data_transformed.append(func(row))

    return data_transformed


def sk(word, alphabet='', p=2):
    return [''.join(f) in word for f in product(alphabet, repeat=p)]


def sk_count(word, alphabet='', p=2):
    return [word.count(''.join(f)) for f in product(alphabet, repeat=p)]
