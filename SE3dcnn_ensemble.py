import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Input, average)
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split



def plot_history(history, result_dir, name):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(name)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(name)))
    plt.close()


def save_history(history, result_dir, name):
    loss=history.history['loss']
    acc=history.history['accuracy']
    val_loss=history.history['val_loss']
    val_acc=history.history['val_accuracy']
    nb_epoch=len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(name)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

def create_3dcnn(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3,3,3), input_shape=(
        input_shape), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3,3,3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.25))

    """model.add(Conv3D(64, kernel_size=(3,3,3), padding='same'))
    model.add(Activation('relu'))"""
    model.add(Conv3D(64, kernel_size=(3,3,3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def main():
    parser=argparse.ArgumentParser(
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
    parser.add_argument('--nmodel', type=int, default=3)
    args=parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    train_dir = 'C:/Users/USER/Downloads/Intro_Deep_Learning-main/Intro_Deep_Learning-main/Examples/Data_Transform_C3D/Data/'
    label_dir = 'C:/Users/USER/Downloads/Intro_Deep_Learning-main/Intro_Deep_Learning-main/Examples/Data_Transform_C3D/Data/'

    train_data = np.load(train_dir + "5_f16_ovp8_gc_mouth.npy")
    label = np.load(label_dir + "5_f16_ovp8_gc_mouth_label.npy")

    print('X_shape:{}\nY_shape:{}'.format(train_data.shape, label.shape))

    X_train, X_test, Y_train, Y_test = train_test_split(train_data, label, test_size=0.2, random_state=43)

    nb_classes = args.nclass
    # Define model
    models=[]
    for i in range(args.nmodel):
        print('model{}:'.format(i))
        models.append(create_3dcnn(train_data.shape[1:], nb_classes))
        models[-1].compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        history = models[-1].fit(X_train, Y_train, validation_data=(
            X_test, Y_test), batch_size=args.batch, epochs=args.epoch, verbose=1, shuffle=True)
        plot_history(history, args.output, i)
        save_history(history, args.output, i)

    model_inputs = [Input(shape=train_data.shape[1:]) for _ in range (args.nmodel)]
    model_outputs = [models[i](model_inputs[i]) for i in range (args.nmodel)]
    model_outputs = average(inputs=model_outputs)
    model = Model(inputs=model_inputs, outputs=model_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #model.summary()

    model.save_weights(os.path.join(args.output, 'ensemble_model.hd5'))

    loss, acc=model.evaluate([X_test]*args.nmodel, Y_test, verbose=0)
    with open(os.path.join(args.output, 'ensemble_result.txt'), 'w') as f:
        f.write('Test loss: {}\nTest accuracy:{}'.format(loss, acc))

    print('merged model:')
    print('Test loss:', loss)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    main()
