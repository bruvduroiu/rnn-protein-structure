def load_data_deprecated():
    '''Loads from txt and cleans data from the UCI protein structure model
    # Returns
        (X_train, y_train), (X_test, y_test): Where each tuple contains lists
        of numpy.ndarrays representing data and targets
    '''
    def _clean(data):
        proteins = data.strip().\
                    replace('<>', '').\
                    replace('<', '').\
                    replace('>', '').\
                    split('end')[:-1]
        
        proteins = list(map(lambda x: x.strip().split('\n'), proteins))
        for i in range(len(proteins)):
            proteins[i] = list(map(lambda x: x.split(' '), proteins[i]))

        for protein in proteins:
            for pair in protein:
                if pair == ['']:
                    protein.remove(pair)

        X = [list(zip(*protein))[0] for protein in proteins]
        y = [list(zip(*protein))[1] for protein in proteins]

        return X, y


    train_data = open('data/protein.train', 'r').read()
    test_data  = open('data/protein.test', 'r').read()

    X_train, y_train = _clean(train_data)
    X_test, y_test = _clean(test_data)

    return (X_train, y_train), (X_test, y_test)

def load_data():
    pass