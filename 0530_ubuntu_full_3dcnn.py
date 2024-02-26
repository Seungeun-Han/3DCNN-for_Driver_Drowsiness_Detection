import argparse
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, Input, Concatenate)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

"""
##########
What's different:
flatten -> concat -> fc -> softmax

Next:
러닝레이트 스케쥴러 추가
testsize, randomstate, verbose, activationfuction등 변경 가능
##########
"""

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1", "/GPU:2", "/GPU:3", "/GPU:4"])

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

def lefteye_Train(train_data):
    print('X_shape:{}'.format(train_data.shape))

    # Define model
    with mirrored_strategy.scope():
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

        model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model.add(Dropout(0.25))

        model.add(Flatten())"""

        inputs = Input(train_data.shape[1:])
        model = Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
            train_data.shape[1:]), padding="same")(inputs)
        model = Activation('relu')(model)
        model = Conv3D(32, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Flatten()(model)

        """model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        # opt = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    model.summary()"""
        return inputs, model

def mouth_Train(train_data):
    print('X_shape:{}'.format(train_data.shape))

    # Define model
    with mirrored_strategy.scope():
        """model1 = Sequential()
        model1.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
            train_data.shape[1:]), padding="same"))
        model1.add(Activation('relu'))
        model1.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
        model1.add(Activation('relu'))
        model1.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model1.add(Dropout(0.25))

        model1.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model1.add(Activation('relu'))
        model1.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model1.add(Activation('relu'))
        model1.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model1.add(Dropout(0.25))

        model1.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model1.add(Activation('relu'))
        model1.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model1.add(Activation('relu'))
        model1.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model1.add(Dropout(0.25))

        model1.add(Flatten())"""
        #model = Sequential()
        inputs = Input(train_data.shape[1:])
        model = Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
            train_data.shape[1:]), padding="same")(inputs)
        model = Activation('relu')(model)
        model = Conv3D(32, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Flatten()(model)

        """model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        # opt = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    model.summary()"""
        return inputs, model

def righteye_Train(train_data):
    print('X_shape:{}'.format(train_data.shape))

    # Define model
    with mirrored_strategy.scope():
        """model2 = Sequential()
        model2.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
            train_data.shape[1:]), padding="same"))
        model2.add(Activation('relu'))
        model2.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
        model2.add(Activation('relu'))
        model2.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model2.add(Dropout(0.25))

        model2.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model2.add(Activation('relu'))
        model2.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model2.add(Activation('relu'))
        model2.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model2.add(Dropout(0.25))

        model2.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model2.add(Activation('relu'))
        model2.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
        model2.add(Activation('relu'))
        model2.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
        model2.add(Dropout(0.25))

        model2.add(Flatten())"""

        inputs = Input(train_data.shape[1:])
        model = Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
            train_data.shape[1:]), padding="same")(inputs)
        model = Activation('relu')(model)
        model = Conv3D(32, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = Conv3D(64, padding="same", kernel_size=(3, 3, 3))(model)
        model = Activation('relu')(model)
        model = MaxPooling3D(pool_size=(3, 3, 3), padding="same")(model)
        model = Dropout(0.25)(model)

        model = Flatten()(model)

        """model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        # opt = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    model.summary()"""
        return inputs, model

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
    parser.add_argument('--type', type=str, default='gc')
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()

    today = str(time.localtime().tm_mon) + str(time.localtime().tm_mday) + '_'

    lefteye_train_data = np.load("../../C3D_predict/C3D_Data/Inputs_f16p8_89892_lefteye.npy")
    lefteye_label = np.load('../../C3D_predict/C3D_Data/Labels_f16p8_lefteye.npy')
    linput, left_model = lefteye_Train(lefteye_train_data)

    mouth_train_data = np.load("../../Dataset/Inputs_f16p8_gc_mouth.npy")
    mouth_label = np.load('../../Dataset/Labels_f16p8_gc_mouth.npy')
    minput, mouth_model = mouth_Train(mouth_train_data)

    righteye_train_data = np.load("../../Dataset/Inputs_f16p8_gc_righteye.npy")
    righteye_label = np.load('../../Dataset/Labels_f16p8_gc_righteye.npy')
    rinput, right_model = righteye_Train(righteye_train_data)

    #inputtt = Input(righteye_train_data.shape[1])

    """left_model.summary()
    mouth_model.summary()
    right_model.summary()"""
    with mirrored_strategy.scope():
        input_model = Concatenate()([left_model, mouth_model, right_model])

        final_model = Dense(512, activation='relu')(input_model)
        final_model = BatchNormalization()(final_model)
        final_model = Dropout(0.5)(final_model)
        final_model = Dense(2, activation='softmax')(final_model)

        # opt = keras.optimizers.Adam(learning_rate=0.01)

        final_model = tf.keras.Model([linput, minput, rinput], final_model)

        final_model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    """X_train, X_test, Y_train, Y_test = train_test_split(
        [lefteye_train_data[:5000], mouth_train_data[:5000], mouth_train_data[:5000]], [lefteye_label[:5000], mouth_label[:5000], righteye_label[:5000]], test_size=0.2, random_state=43)


    history = final_model.fit([lefteye_train_data[:5000], mouth_train_data[:5000], mouth_train_data[:5000]], righteye_label[:5000],  validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)

    loss, acc = final_model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)"""

    history = final_model.fit([lefteye_train_data[:5000], mouth_train_data[:5000], righteye_train_data[:5000]], righteye_label[:5000]
                              , validation_data=([lefteye_train_data[5000:5500], mouth_train_data[5000:5500], righteye_train_data[5000:5500]], righteye_label[5000:5500])
                              , batch_size=args.batch, epochs=args.epoch, verbose=1, shuffle=True)

    loss, acc = final_model.evaluate([lefteye_train_data[5000:5500], mouth_train_data[5000:5500], righteye_train_data[5000:5500]], righteye_label[5000:5500], verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    final_model.save_weights(os.path.join(args.output, today + 'f16_ovp8_' + args.type + '_' + '_model.h5'))

    plot_history(history, args.output, today, args.count, args.type)
    save_history(history, args.output, today, args.count, args.type)
