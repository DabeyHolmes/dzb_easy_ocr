import os
import time

import cv2
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from fast_read_data import ChineseWrittenChars
import split_picture as sp
from keras.models import load_model
from PIL import Image
import numpy as np
import pickle
import struct

chars = ChineseWrittenChars()
chars.test.use_rotation = False
chars.test.use_filter = False

lb = LabelBinarizer()
lb.fit(chars.generate_char_list())
number_of_classes = 3755


def build_model():
    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape=(64, 64, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3755))

    model.add(Activation('softmax'))
    return model


def training(X_train, y_train):
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=12)  # the more epoch the better
    model.save('model.h5')


def model_test(X_test, y_test):
    # load model
    from keras.models import load_model
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loaded_model_score = model.evaluate(X_test, y_test)
    print('test accuracy: ', loaded_model_score[1])  # the 0-th element is loss, the 1st element is accuracy
    print(model.metrics)


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def load_picture(path_test_image):
    # color = (0, 0, 255)
    peek_ranges, vertical_peek_ranges2d, image_color = sp.get_font_face_peek_ranges(path_test_image)
    mycount = 0
    image_list = []
    for i, peek_range in enumerate(peek_ranges):
        for (j, vertical_range) in enumerate(vertical_peek_ranges2d[i]):
            x = vertical_range[0]
            y = peek_range[0]

            w = vertical_range[1] - x
            h = peek_range[1] - y
            image = image_color[y - 2:y + h + 2, x - 2:x + w + 2]
            path = "temp/img" + str(mycount) + ".png"
            image_list.append(path)
            cv2.imwrite(path, image)

            # pt1 = (x, y)
            # pt2 = (x + w, y + h)
            # cv2.rectangle(image_color, pt1, pt2, color)
            mycount += 1
    #
    # cv2.imshow('image', image_color)
    # cv2.waitKey(0)
    return image_list


def more_pic(path):
    del_file('temp')
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    im_list = load_picture(path)
    cc = 0
    for impath in im_list:
        im = Image.open(impath)
        im = im.resize((64, 64), Image.ANTIALIAS)
        im = im.convert("L")
        im.save('transform/' + str(cc) + '.png')
        data = im.getdata()
        data = np.array(data, dtype='float') / 255.0
        data1 = data.reshape([1, 64, 64, 1])
        Y = model.predict(data1)
        y = lb.inverse_transform(Y)
        print(struct.pack('>H', int(y)).decode('gb2312'))
        cc += 1


def one_pic(path):
    del_file('temp')
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    im = Image.open(path)
    im = im.resize((64, 64), Image.ANTIALIAS)
    im = im.convert("L")
    a = [0] * 200 + [1] * 56
    # im = im.point(a, '1')
    # im.save('transform/' + str(0) + '.png')
    data = im.getdata()
    data = np.array(data, dtype='float') / 255.0
    data1 = data.reshape([1, 64, 64, 1])
    Y = model.predict(data1)
    y = lb.inverse_transform(Y)
    print(struct.pack('>H', int(y)).decode('gb2312'))



if __name__ == '__main__':
     # one_pic('test/try.jpg')
     more_pic('test/jiancha.png')
