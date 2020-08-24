from sklearn.preprocessing import OneHotEncoder

ALPHABET = '*'

def transformation(row, verbose=False, array=True  ):
    if array:
        string = row[0]
    else:
        string = row
    if verbose:
        if array:
            print(string + " => " + row[0])
        else:
            print(string + " => " + row)

    return row


def one_hot_encoding(row, verbose=False, onehot_encoder=OneHotEncoder(), array=True):
    if array:
        data = list(row[0])
    else:
        data = list(row)
    tmp = []

    for k in range(len(data)):
        encoding = onehot_encoder.transform([[data[k]]])[0]

        tmp.append(list(encoding))
        if verbose:
            print(data[k] + " => " + str(encoding))

    return tmp

if __name__ == "__main__":
    #print(string.printable)
    val = transformation(['aAcC12 3{4)[]}( $%+/*-^<'], True)
    assert val == ['aAcC12 3{4)[]}( $%+/*-^<'], "Encoding is WRONG!"
    print(one_hot_encoding(val, True))
