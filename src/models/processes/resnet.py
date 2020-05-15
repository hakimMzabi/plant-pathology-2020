'''
ResNet process to generate models for the CIFAR-10 dataset.
'''
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Add,
    Activation,
    Input,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout
)

# training parameters
from src.cifar10 import Cifar10
from src.helper import Helper


class Residual:
    @staticmethod
    def block(input_data, filters, conv_size):
        x = Conv2D(filters, conv_size, activation='relu', padding="same")(input_data)
        x = BatchNormalization()(x)
        x = Conv2D(filters, conv_size, activation=None, padding="same")(x)
        x = BatchNormalization()(x)
        x = Add()([x, input_data])
        x = Activation("relu")(x)
        return x


def create_model(n_resblocks=10, h_filters=64, l_filters=32, dropout=0.5, n_neurons=256, k_size=3):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(l_filters, k_size, activation='relu')(inputs)
    x = Conv2D(h_filters, k_size, activation='relu')(x)
    x = MaxPooling2D(k_size)(x)

    for i in range(n_resblocks):
        x = Residual.block(x, h_filters, k_size)

    x = Conv2D(h_filters, k_size, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_neurons, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    return model
