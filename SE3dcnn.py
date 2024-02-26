import argparse
import videoto3d
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, Input)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import tensorflow as tf

def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '5_f16_ovp8_gc_mouth_model_accuracy2.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '5_f16_ovp8_gc_mouth_model_loss2.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '5_f16_ovp8_gc_mouth_result2.txt'), 'w') as fp:
        fp.write('epoch\tloss\taccuracy\tval_loss\tval_accuracy\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--videos', type=str, default='',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--output', type=str, default='./result/')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=16)
    args = parser.parse_args()


    train_dir = 'D:/Dataset/NTHU-DDD-npy/'
    label_dir = 'D:/Dataset/NTHU-DDD-npy/'

    train_data = np.load(train_dir + "Inputs_f16p8_gc_righteye.npy")
    label = np.load(label_dir + "Labels_f16p8_gc_righteye.npy")

    print('X_shape:{}\nY_shape:{}'.format(train_data.shape, label.shape))

    # Define model
    """model = Sequential()
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

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))"""
    model = tf.keras.Sequential([
        Input(train_data.shape[1:]),
        Conv3D(32, kernel_size=(3, 3, 3), input_shape=(train_data.shape[1:]), padding="same"),
        Activation('relu'),
        Dropout(0.25),
        MaxPooling3D(pool_size=(3, 3, 3), padding="same"),
        Dropout(0.25),
        Conv3D(64, padding="same", kernel_size=(3, 3, 3)),
        Activation('relu'),
        MaxPooling3D(pool_size=(3, 3, 3), padding="same"),
        Dropout(0.25),
        MaxPooling3D(pool_size=(3, 3, 3), padding="same"),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(loss=categorical_crossentropy,
                  optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(train_data[:100], label[:100], test_size=0.2, random_state=43)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    model.save_weights(os.path.join(args.output, 'f16_ovp8_gc_righteye_model.hd5'))

    plot_history(history, args.output)
    save_history(history, args.output)
