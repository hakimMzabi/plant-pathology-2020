'''
RNN process to generate models for the CIFAR-10 dataset.
'''

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy


def create_model(n_layers, optimizer, n_neurons, dropout):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if dropout is not None:
                model.add(LSTM(n_neurons, dropout=dropout, return_sequences=True, input_shape=(32, 96)))
            else:
                model.add(LSTM(n_neurons, return_sequences=True, input_shape=(32, 96)))
        else:
            if i == n_layers - 1:
                if dropout is not None:
                    model.add(LSTM(n_neurons, dropout=dropout))
                else:
                    model.add(LSTM(n_neurons))
            else:
                if dropout is not None:
                    model.add(LSTM(n_neurons, dropout=dropout, return_sequences=True))
                else:
                    model.add(LSTM(n_neurons, return_sequences=True))
    model.add(Dense(10, activation="sigmoid"))
    model.compile(
        optimizer=optimizer,
        loss=sparse_categorical_crossentropy,
        metrics=[sparse_categorical_accuracy]
    )

    return model


if __name__ == "__main__":
    model = create_model(
        2, "adam", 64, None
    )
    from src.helper import Helper
    from src.cifar10 import Cifar10

    model.summary()
    cifar10 = Cifar10(dim=2)

    Helper().fit(
        model,
        cifar10.x_train,
        cifar10.y_train,
        1024,
        100,
        (cifar10.x_test, cifar10.y_test),
        "rnn",
        std_logs=True
    )