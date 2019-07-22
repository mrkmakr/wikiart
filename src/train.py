from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

NUMBER_OF_CLASSES = 10


def CNN(hx=128, hy=128):
    model = Sequential()
    model.add(Conv2D(128, 3, input_shape=(hx, hy, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))

    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))

    adam = Adam(lr=1e-5)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])
    return model


if __name__ == "__main__":
    X, y = np.load('../train/X.npy'), np.load('../train/y.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

    Y_train = np_utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NUMBER_OF_CLASSES)

    model = CNN()
    history = model.fit(
            X_train,
            Y_train,
            batch_size=64,
            nb_epoch=100,
            verbose=1,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, verbose=1)])

    #モデルの表示
    model.summary()

    #評価
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:',score[0])
    print('Test accuracy:',score[1])

    json_string = model.to_json()
    with open('../cnn.json', "w") as f:
        f.write(json_string)

    model.save_weights('../cnn.h5')