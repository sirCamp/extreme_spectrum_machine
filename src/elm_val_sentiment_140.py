import time

from MKLpy.metrics.pairwise import spectrum_embedding
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.data_utils import load_csv_data_filter
from sklearn.random_projection import GaussianRandomProjection

files = ['sentiment_140_']

data_path = '../data/'

def input_to_hidden(x, w):
    a = np.dot(x, w)
    a = np.maximum(a, 0, a) # ReLU
    return a

def get_weights(input_lenght, hidden_units):


    w = np.random.normal(size=[input_lenght, hidden_units])
    #print('Input Weight shape: {shape}'.format(shape=w.shape))
    return w


def train(x_train, y_train, w):
    X = input_to_hidden(x_train, w)
    Xt = np.transpose(X)

    try:
        out = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
    except:
        out = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, y_train))
    #print('Output weights shape: {shape}'.format(shape=Wout.shape))

    return out


def predict(x, w, out):
    x = input_to_hidden(x, w)
    y = np.dot(x, out)
    return y


def to_matrix(X, dictionary=None):
    if not dictionary:
        features = {f for x in X for f in x.keys()}
        dictionary = {f: i for i, f in enumerate(features)}
    V = np.zeros((len(X), len(dictionary)))
    for row, x in enumerate(X):
        for k, v in x.items():
            if k in dictionary:
                V[row, dictionary[k]] = v
    return V, dictionary

for file in files:


    data, labels = load_csv_data_filter(
        path=data_path, filename=file)

    maps = np.load( data_path + file + "train_test_val_splits.npy")
    maps = maps[()]


    data = data[maps['dataset_index']]
    labels = labels[maps['dataset_index']]

    tr_data = data[maps['train_train_index']]
    tt_data = data[maps['test_index']]

    o_train_data_label = labels[maps['train_train_index']]
    test_data_label = labels[maps['test_index']]

    onehot = OneHotEncoder(sparse=False)
    onehot.fit(o_train_data_label.reshape(len(o_train_data_label), 1))
    o_train_data_label = onehot.transform(o_train_data_label.reshape(len(o_train_data_label), 1))
    test_data_label = onehot.transform(test_data_label.reshape(len(test_data_label), 1))

    train_data = []
    for t in tr_data:
        train_data.append(t)
    o_train_data = np.array(train_data)

    test_data = []
    for t in tt_data:
        test_data.append(t)
    test_data = np.array(test_data)

    train_data = o_train_data[maps['train_index']]
    val_data = o_train_data[maps['val_index']]
    train_data_label = o_train_data_label[maps['train_index']]
    val_data_label = o_train_data_label[maps['val_index']]


    accuracies = []
    stds = []
    components = []
    size = 0
    best_acc = 0
    pspec = 0
    accuracies_val = ""
    best_w = None
    for p in [1, 2, 3, 4, 5, 6]:

        Ep_tr = spectrum_embedding(train_data, p=p)

        Ep_val = spectrum_embedding(val_data, p=p)
        Ep_te = spectrum_embedding(test_data, p=p)

        Ep_tr, d = to_matrix(Ep_tr)
        Ep_te, _ = to_matrix(Ep_te, d)
        Ep_val, _ = to_matrix(Ep_val, d)

        for j in [50, 100, 200, 500, 700, 1000]:

            transformer = GaussianRandomProjection(n_components=j, random_state=42)
            Xtr = transformer.fit_transform(Ep_tr)
            Xvl = transformer.fit_transform(Ep_val)

            w = get_weights(Xtr.shape[1], j)

            out = train(Xtr, train_data_label, w)
            y = predict(Xvl,w,out)

            correct = 0
            total = y.shape[0]
            for i in range(total):
                predicted = np.argmax(y[i])
                test = np.argmax(val_data_label[i])
                correct = correct + (1 if predicted == test else 0)

            scores = float(correct)/float(total)
            print("Accuracy val: " + str(scores)+ "p: "+str(p)+" size: "+str(j))
            accuracies_val += "Accuracy val: " + str(scores)+ "p: "+str(p)+" size: "+str(j)+"\n"
            if scores > best_acc:
                best_acc = scores
                size = j
                pspec = p
                best_w = np.copy(w)


        components.append(size)


    Ep_tr = spectrum_embedding(o_train_data, p=pspec)
    Ep_te = spectrum_embedding(test_data, p=pspec)

    Ep_tr, d = to_matrix(Ep_tr)
    Ep_te, _ = to_matrix(Ep_te, d)

    internal_accuracies = []
    times = []
    w = np.copy(best_w)
    np.save(data_path + file + "_elm_weights.npy", w)
    for i in range(5):
        transformer = GaussianRandomProjection(n_components=size, random_state=i)

        Xtr = transformer.fit_transform(Ep_tr)
        Xte = transformer.fit_transform(Ep_te)

        start_time = time.time()

        out = train(Xtr, o_train_data_label, w)
        y = predict(Xte, w, out)

        correct = 0
        total = y.shape[0]
        for i in range(total):
            predicted = np.argmax(y[i])
            test = np.argmax(test_data_label[i])
            correct = correct + (1 if predicted == test else 0)

        scores = float(correct)/float(total)
        end_time = time.time()

        print("Accuracy: " + str(scores))
        internal_accuracies.append(scores)
        times.append(end_time-start_time)

    internal_accuracies = np.array(internal_accuracies)

    times = np.array(times)
    acc = internal_accuracies.mean()
    st = internal_accuracies.std()
    tm = times.mean()
    tm_std = times.std()

    print(accuracies_val)
