from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def _add_batch_conv_2d(model, ch, ksize, stride=1, pad=0, activation='relu', input_shape=None, border_mode='valid'):
    if input_shape is None:
        model.add(Convolution2D(ch, ksize, ksize, border_mode=border_mode))
    else:
        model.add(Convolution2D(ch, ksize, ksize, border_mode=border_mode, input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation(activation))

def cnn_bn():
    model = Sequential()
    _add_batch_conv_2d(model, 64, 5, input_shape=(32, 32, 3), border_mode='same')
    model.add(MaxPooling2D(pool_size=(2,2)))
    _add_batch_conv_2d(model, 64, 3, border_mode='same')
    model.add(MaxPooling2D(pool_size=(2,2)))
    _add_batch_conv_2d(model, 128, 3, border_mode='same')
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def vgg():
    model = Sequential()
    _add_batch_conv_2d(model, 64, 3, input_shape=(32, 32, 3), border_mode='same')
    _add_batch_conv_2d(model, 64, 3, border_mode='same')
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    _add_batch_conv_2d(model, 128, 3, border_mode='same')
    _add_batch_conv_2d(model, 128, 3, border_mode='same')
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    _add_batch_conv_2d(model, 256, 3, border_mode='same')
    _add_batch_conv_2d(model, 256, 3, border_mode='same')
    _add_batch_conv_2d(model, 256, 3, border_mode='same')
    _add_batch_conv_2d(model, 256, 3, border_mode='same')
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model
