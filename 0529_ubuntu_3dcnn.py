import argparse
import videoto3d
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

"""
##########
What's different:
데이터셋을 gc_mouth로
gpu 여러개 쓰는 코드 추가

Next:
러닝레이트 스케쥴러 추가
testsize, randomstate, verbose, activationfuction등 변경 가능
##########
"""

def plot_history(history, result_dir, today, count, type):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '0' + today + type + '_model_accuracy' + str(count) + '.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '0' + today + type + '_model_loss' + str(count) + '.png'))
    plt.close()


def save_history(history, result_dir, today, count, type):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '0' + today + type + '_result' + str(count) + '.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=112)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--output', type=str, default='./result/')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--part', type=str, default='mouth')
    parser.add_argument('--type', type=str, default='gc')
    parser.add_argument('--count', type=int, default=0)
    args = parser.parse_args()

    today = str(time.localtime().tm_mon) + str(time.localtime().tm_mday) + '_'

    train_data = np.load("./dataset/concat_f16_ovp8_gc_mouth.npy")
    label = np.load('./dataset/concat_f16_ovp8_gc_mouth_label.npy')

    print('X_shape:{}\nY_shape:{}'.format(train_data.shape, label.shape))

    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1", "/GPU:2", "/GPU:3"])

    # Define model
    #with mirrored_strategy.scope():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        train_data.shape[1:]), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    #opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(
        train_data, label, test_size=0.2, random_state=43)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)

    loss, acc = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    model.save_weights(os.path.join(args.output, today + 'f16_ovp8_' + args.type + '_' + args.part + '_model.h5'))

    plot_history(history, args.output, today, args.count, args.type)
    save_history(history, args.output, today, args.count, args.type)
