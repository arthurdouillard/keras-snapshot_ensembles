import os
import glob

from keras.datasets import mnist
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils

from snapshot import Snapshot

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def format(x, y):
    x = x.reshape((x.shape[0], 28, 28, 1))
    x = x.astype('float32') / 255
    y = np_utils.to_categorical(y, 10)
    return x, y

x_train, y_train = format(x_train, y_train)
x_test, y_test = format(x_test, y_test)


def get_model(tensor):
    x = Conv2D(32, (3,3), activation='relu', padding='same', use_bias=False)(tensor)
    x = Conv2D(32, (3,3), activation='relu', padding='same', use_bias=False)(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x_out = Dense(10, activation='softmax')(x)

    return Model(inputs=tensor, outputs=x_out)

model = get_model(Input(shape=(28, 28, 1)))
model.compile(
    optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cbs = [Snapshot('snapshots', nb_epochs=6, verbose=1, nb_cycles=2)]

model.fit(
    x=x_train, y=y_train,
    verbose=1, batch_size=124, epochs=5,
    callbacks=cbs
)

del model

# Loading the ensemble
print('Loading ensemble...')

keep_last = 2

def load_ensemble(folder, keep_last=None):
    paths = glob.glob(os.path.join(folder, 'weights_cycle_*.h5'))
    print('Found:', ', '.join(paths))
    if keep_last is not None:
        paths = sorted(paths)[-keep_last:]
    print('Loading:', ', '.join(paths))

    x_in = Input(shape=(28, 28, 1))
    outputs = []

    for i, path in enumerate(paths):
        m = get_model(x_in)
        m.load_weights(path)
        outputs.append(m.output)

    shape = outputs[0].get_shape().as_list()
    x = Lambda(lambda x: K.mean(K.stack(x, axis=0), axis=0),
               output_shape=lambda _: shape)(outputs)
    model = Model(inputs=x_in, outputs=x)
    return model


model = load_ensemble('snapshots')
model.compile(
    optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

metrics = model.evaluate(x_test, y_test)
print(metrics)