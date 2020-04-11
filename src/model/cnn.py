import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.estimator import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parametres
BATCH_SIZE=32
IMG_WIDTH=150
IMG_HEIGHT=150
EPOCHS=15


# create a data generator
data_gen = ImageDataGenerator(rescale=1./255,
                rotation_range=45,
                width_shift_range=.15,
                height_shift_range=.15,
                horizontal_flip=True,
                zoom_range=0.5
                )

data_test_gen = ImageDataGenerator(rescale=1./255)

# load and iterate training dataset
train_gen = data_gen.flow_from_directory(directory='./data/train',
                                         target_size=(IMG_WIDTH,IMG_HEIGHT),
                                         class_mode='categorical',
                                         color_mode='rgb',
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         seed=42)
# load and iterate training dataset
validation_gen = data_test_gen.flow_from_directory(directory='./data/validation',
                                         target_size=(IMG_WIDTH,IMG_HEIGHT),
                                         class_mode='categorical',
                                         color_mode='rgb',
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         seed=42)
#load and iterate test dataset
test_gen = data_gen.flow_from_directory(directory='./data/test',
                                        target_size=(IMG_WIDTH,IMG_HEIGHT),
                                        class_mode='categorical',
                                        color_mode='rgb',
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        seed=42)

# plot some images
sample_training, _ = next(train_gen)

def plotImage(images):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        print(img.shape)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImage((sample_training[:5]))


def lr_optimizer(epoch):
    lr = 1e-3
    if (epoch == 30):
        lr *= 1e-1
    elif(epoch == 70):
        lr *= 1e-1
    else:
        lr *= 0.5e-1


def create_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', activation=relu, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3, 3), padding='same', activation=relu))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(512, activation=relu))  # , kernel_regularizer=l1_l2(l2=0.01) ))
    model.add(Dropout(0.3))

    model.add(Dense(4, activation=softmax))

    # compile the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


def create_model2():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', activation=relu, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Conv2D(16, (3, 3), padding='same', activation=relu))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3, 3), padding='same', activation=relu))
    model.add(Conv2D(32, (3, 3), padding='same', activation=relu))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(Conv2D(64, (3, 3), padding='same', activation=relu))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, activation=relu))  # ,  kernel_regularizer=l1_l2(l2=0.01) ))
    model.add(Dropout(0.4))

    model.add(Dense(4, activation=softmax))

    # compile the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model


# summary
model = create_model()
model.summary()

#train_model

history=model.fit_generator(generator=train_gen,
                    steps_per_epoch=train_gen.n//train_gen.batch_size,
                    epochs=120,
                    validation_data=validation_gen,
                    validation_steps=validation_gen.n//validation_gen.batch_size)


# evaluating the model
model.evaluate_generator(generator=validation_gen,
                         steps=validation_gen.n//validation_gen.batch_size)

#predict the output
test_gen.reset()
pred = model.predict_generator(test_gen,
                               steps=test_gen.n//test_gen.batch_size,
                               verbose=1)

#reset the test generator
predicted_class_indices = np.argmax(pred,axis=1)

# map predictions
tmp_filenames = [id[5:20] for id in test_gen.filenames]
filenames = []
for file in tmp_filenames:
    f = file.split('.')
    filenames.append(f[0])

df = pd.DataFrame({'image_id': filenames})
df = df.join(pd.DataFrame(data=pred, columns=train_gen.class_indices.keys()))

df.head(10)

df.to_csv("results.csv", index=False)