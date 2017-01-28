import argparse
import numpy as np
import six
from six.moves import cPickle as pickle

import cifar_model

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 dataset trainer')
    parser.add_argument('--model', '-m', type=str, default='vgg', choices=['cnn', 'cnnbn', 'vgg', 'residual', 'identity_mapping'],
                        help='Model name')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='Mini batch size')
    parser.add_argument('--dataset', '-d', type=str, default='dataset/image.pkl',
                        help='Dataset image pkl file path')
    parser.add_argument('--label', '-l', type=str, default='dataset/label.pkl',
                        help='Dataset label pkl file path')
    parser.add_argument('--prefix', '-p', type=str, default=None,
                        help='Prefix of model parameter files')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--save_iter', type=int, default=0,
                        help='Iteration interval to save model parameter file.')
    parser.add_argument('--lr_decay_iter', type=int, default=100,
                        help='Iteration interval to decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate for SGD')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Initial alpha for Adam')
    parser.add_argument('--res_depth', type=int, default=18,
                        help='Depth of Residual Network')
    parser.add_argument('--skip_depth', action='store_true',
                        help='Use stochastic depth in Residual Network')
    parser.add_argument('--swapout', action='store_true',
                        help='Use swapout')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    args = parser.parse_args()

    batch_size = args.batch_size
    epoch_num = args.epoch
    class_num = 10

    print('loading dataset...')
    with open(args.dataset, 'rb') as f:
        images = pickle.load(f)
        index = np.random.permutation(len(images['train']))
        train_index = index[:-5000]
        valid_index = index[-5000:]
        x_train = images['train'][train_index].reshape((-1, 3, 32, 32))
        x_train = x_train.transpose((0, 2, 3, 1))
        x_valid = images['train'][valid_index].reshape((-1, 3, 32, 32))
        x_valid = x_valid.transpose((0, 2, 3, 1))
        x_test = images['test'].reshape((-1, 3, 32, 32))
        x_test = x_test.transpose((0, 2, 3, 1))
    with open(args.label, 'rb') as f:
        labels = pickle.load(f)
        y_train = labels['train'][train_index]
        y_valid = labels['train'][valid_index]
        y_test = labels['test']

    y_train = np_utils.to_categorical(y_train, class_num)
    y_valid = np_utils.to_categorical(y_valid, class_num)
    y_test = np_utils.to_categorical(y_test, class_num)

    if args.model == 'cnn_bn':
        model = cifar_model.cnn_bn()
    elif args.model == 'vgg':
        model = cifar_model.vgg()
    else:
        raise 'Not supported model'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), nb_epoch=epoch_num, verbose=1, samples_per_epoch=len(x_train), validation_data=(x_valid, y_valid))

#    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch_num,
#              verbose=1, validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
