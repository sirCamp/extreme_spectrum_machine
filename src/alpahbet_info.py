
import numpy as np

from src.data_utils import data_encoding_transformation, load_csv_data_filter,load_csv_data

data_path = "../data/"
files= ['sentiment_140','twitter_sa','BioNLP13CG_cc','BioNLP13CG_chemical','BioNLP13CG_species','BioNLP13PC_cc','BioNLP13PC_chemical']
for file in files:

    data, labels = load_csv_data(
        path=data_path, filename=file)

    mapps = np.load(data_path + file + "train_test_val_splits.npy")
    mapps = mapps[()]

    train_data = data[mapps['train_train_index']]
    test_data = data[mapps['test_index']]

    lenghts = []
    chars = np.array([])
    for r in train_data:
        lenghts.append(len(r[1]))
        chars = np.concatenate((chars, np.unique(np.array(list(r[1])))))

    lengths = np.array(lenghts)

    chars = np.unique(chars)

    print("mean: "+str(lengths.mean()))
    print("std: "+str(lengths.std()))
    print("max: " + str(lengths.max()))
    print("unique:" +str(len(chars)))
    print(chars)