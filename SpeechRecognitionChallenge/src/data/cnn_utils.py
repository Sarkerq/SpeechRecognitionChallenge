import numpy as np
from PIL import Image
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_mobilenet
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Conv2D, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import adam
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, LSTM

from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D


def dump_all_to_csv():
    train_paths = ['..\\..\\data\\raw\\train_images\\' + str(x) + '.png' for x in range(1, 50001)]
    images_to_csv(train_paths, '..\\..\\data\\raw\\train.csv')


def dump_all_test_to_csv():
    for i in range(0, 5):
        train_paths = ['..\\..\\data\\raw\\test_images\\' + str(x) + '.png' for x in
                       range(1 + 50000 * i, 50001 + 50000 * i)]
        images_to_csv(train_paths, f'..\\..\\data\\raw\\test{i}.csv')


def images_to_csv(img_paths, csv_file):
    images = np.array(
        [load_image_to_array(path) for path in img_paths])
    save_to_csv(images, csv_file)


def save_to_csv(data, filepath):
    pd.DataFrame(data).to_csv(filepath)


def load_image_to_array(filename):
    return load_image_to_3d_array(filename).flatten()


def load_image_to_3d_array(filename):
    im = Image.open(filename)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]

    arr = np.array([[pix[x, y] for x in range(width)] for y in range(height)])
    return arr


def output_number_to_class(number):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return classes[number]


def output_class_to_number(klass):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return classes.index(klass)


def preprocess_and_save_to_csv(array, arch, path, inplace=True, scaler=None):
    if not inplace:
        array = array.copy()
    if arch == "VGG19":
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        array = preprocess_vgg19(array).reshape(-1, 32, 32, 3)

    elif arch == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        array = preprocess_resnet50(array).reshape(-1, 32, 32, 3)

    else:
        return None
    array = base_model.predict(array).reshape(-1, 512)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(array)

    array = scaler.transform(array)

    save_to_csv(array, path)
    return [array, scaler]


def separate_test_dev(x_raw, y_raw, train_size, dev_size):
    data_size = y_raw.shape[0]
    x_dev = x_raw[data_size - dev_size:]
    x_train = x_raw[:train_size]

    y_dev = y_raw[data_size - dev_size:]
    y_train = y_raw[:train_size]
    return x_train, y_train, x_dev, y_dev


def separate_train_val_ens(x_raw, y_raw, sizes):
    train_end = sizes[0]
    val_end = sizes[1] + sizes[0]

    x_train = x_raw[:train_end]
    x_val = x_raw[train_end:val_end]
    x_ens = x_raw[val_end:]

    y_train = y_raw[:train_end]
    y_val = y_raw[train_end:val_end]
    y_ens = y_raw[val_end:]
    return x_train, y_train, x_val, y_val, x_ens, y_ens


def z_norm(x_train, x_val, x_ens = None, x_test = None):
    print("FUP")
    mean = np.mean(x_train, axis=(0, 1, 2))
    std = np.std(x_train, axis=(0, 1, 2))
    x_train = (x_train - mean) / (std + 1e-7)
    x_val = (x_val - mean) / (std + 1e-7)
    if x_ens != None:
        x_ens = (x_ens - mean) / (std + 1e-7)
    if x_test != None:
        x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_val, x_ens, x_test


def preprocess_res(x_train, x_val, x_ens = None, x_test = None):
    x_train = preprocess_resnet50(x_train)
    x_val = preprocess_resnet50(x_val)
    if x_ens != None:
        x_ens = preprocess_resnet50(x_ens)
    if x_test != None:
        x_test = preprocess_resnet50(x_test)
    return x_train, x_val, x_ens, x_test


def preprocess_mob(x_train, x_val, x_ens = None, x_test = None):
    x_train = preprocess_mobilenet(x_train)
    x_val = preprocess_mobilenet(x_val)
    if x_ens != None:
        x_ens = preprocess_mobilenet(x_ens)
    if x_test != None:
        x_test = preprocess_mobilenet(x_test)
    return x_train, x_val, x_ens, x_test


def NiNBlock(kernel, mlps, strides):
    def inner(x):
        l = Conv2D(mlps[0], kernel, strides=strides, padding='same')(x)
        l = Activation('relu')(l)
        for size in mlps[1:]:
            l = Conv2D(size, 1, strides=[1, 1])(l)
            l = Activation('relu')(l)
        return l

    return inner


def get_nin_model_by_input(img_rows, img_cols, channels):
    img = Input(shape=(img_rows, img_cols, channels))
    l1 = NiNBlock(5, [192, 160, 96], [1, 1])(img)
    l1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(l1)
    l1 = Dropout(0.7)(l1)

    l2 = NiNBlock(5, [192, 192, 192], [1, 1])(l1)
    l2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(l2)
    l2 = Dropout(0.7)(l2)

    l3 = NiNBlock(3, [192, 192, 10], [1, 1])(l2)

    l4 = GlobalAveragePooling2D()(l3)
    l4 = Activation('softmax')(l4)

    model = Model(inputs=img, outputs=l4)
    return model


def get_nin_model(x_train):
    model = get_nin_model_by_input(32, 32, 3)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def get_lstm_model():
    model = Sequential()
    model.add(LSTM(110, return_sequences=True, input_shape=(99,161)))
    model.add(BatchNormalization())
    model.add(LSTM(110, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(110, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(110, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(110, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(110))
    model.add(BatchNormalization())
    model.add(Dense(110))
    model.add(BatchNormalization())
    model.add(Dense(11))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',  optimizer='adam',metrics=['categorical_accuracy'])


def get_aml_model(x_train):
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_zoo_model(x_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(11))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def combine_with_base_model(base_model):
    model = Sequential()
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    combined_model = Sequential()
    combined_model.add(base_model)
    combined_model.add(BatchNormalization())
    combined_model.add(model)
    combined_model.layers[0].trainable = True

    combined_model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
    return combined_model


def get_resnet_model(x_train):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    return combine_with_base_model(base_model)


def get_mobilenet_model(x_train):
    return combine_with_base_model(MobileNet(include_top=False, weights=None, input_shape=(32, 32, 3)))


def get_datagen():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


def get_pred(model, X):
    return model.predict(X)
