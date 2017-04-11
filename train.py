"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import data_parser
import numpy as np
from keras.optimizers import Adadelta, Adam, rmsprop
from sklearn.metrics import mean_squared_error

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_dbtt():
    data = data_parser.parse("DBTT_Data22.csv")
    data_lwr = data_parser.parse("CD_LWR_clean8.csv")
    X = ["N_log(eff fl p =.05)", "N_log(eff fl p =.4)", "N_log(eff fl p =.5)", "N(Cu)", "N(Ni)", "N(Mn)", "N(P)",
         "N(Si)", "N( C )", "N_log(eff fl p =.1)", "N_log(eff fl p =.2)", "N_log(eff fl p =.3)", "N(Temp)"]
    Y = "CD delta sigma"
    data.set_x_features(X)
    data.set_y_feature(Y)
    data_lwr.set_y_feature(Y)
    data_lwr.set_x_features(X)
    data.add_exclusive_filter("Alloy", '=', 29)
    data.add_exclusive_filter("Alloy", '=', 8)
    data.add_exclusive_filter("Alloy", '=', 1)
    data.add_exclusive_filter("Alloy", '=', 2)
    data.add_exclusive_filter("Alloy", '=', 14)

    data_lwr.add_exclusive_filter("Alloy", '=', 29)
    data_lwr.add_exclusive_filter("Alloy", '=', 14)
    x_test = np.array(data_lwr.get_x_data())
    y_test = np.array(data_lwr.get_y_data())
    x_train = np.array(data.get_x_data())
    y_train = np.array(data.get_y_data())
    #print("Training with", np.shape(y_train)[0], "data points")

    nb_classes = -1
    batch_size = np.shape(y_train)[0]
    input_shape = (13,)

    # normalize y columns
    y_train = y_train/758.92
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    learning_rate = network['learning_rate']

    model = Sequential()
    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            print(nb_neurons)
            model.add(Dense(units=nb_neurons, activation=activation, input_shape=input_shape))
        else:
            print(nb_neurons)
            model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(0.2))  # hard-coded dropout


    # Output layer.
    if(nb_classes == -1):
        model.add(Dense(1, activation='linear'))
        ADAM = Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=ADAM)
    else:
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()
    elif dataset == 'dbtt':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_dbtt()

    model = compile_model(network, nb_classes, input_shape)

    if dataset == 'dbtt':
        model.fit(x_train, y_train, epochs=10, batch_size=1406, verbose=0)
        y_predict = model.predict(x_test) * 758.92  # todo way to not hardcode this?
        rms = np.sqrt(mean_squared_error(y_test, y_predict))
        print(rms)
        return rms
    else:
        model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

        score = model.evaluate(x_test, y_test, verbose=0)

        return score[1]  # 1 is accuracy. 0 is loss.
