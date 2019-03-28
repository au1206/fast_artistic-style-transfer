import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Activation
from keras.losses import mean_squared_error



def get_full_network(height, width):

    inputs = Input(shape=(height, width, 3))
    #1
    x = Conv2D(32, (9,9), strides=1, padding='same',input_shape=(height,width,3))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #2
    x = Conv2D(64, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #3
    x = Conv2D(128, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(0,5):
        x = Conv2D(128, (3,3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3,3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2DTranspose(64,(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32,(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(3, (9,9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    out0 = Activation('relu')(x)    #output_image

    vggnet = VGG16(include_top=False, input_shape=(height, width, 3))

    for layer in vggnet.layers:
        layer.trainable = False

    x = vggnet.layers[1](out0)  # block1_conv1
    out1 = vggnet.layers[2](x)  # block1_conv2
    x = vggnet.layers[3](out1)  # block1_pool

    x = vggnet.layers[4](x)  # block2_conv1
    out2 = vggnet.layers[5](x)  # block2_conv2
    x = vggnet.layers[6](out2)  # block2_pool

    x = vggnet.layers[7](x)  # block3_conv1
    x = vggnet.layers[8](x)  # block3_conv2
    out3 = vggnet.layers[9](x)  # block3_conv3
    x = vggnet.layers[10](out3)  # block3_pool

    x = vggnet.layers[11](x)  # block3_conv1
    x = vggnet.layers[12](x)  # block3_conv2
    out4 = vggnet.layers[13](x)  # block3_conv3

    model = Model(inputs=inputs, outputs=[out0, out1, out2, out3, out4, out3])

    return model


def get_vgg(height, width):
    vggnet = VGG16(include_top=False, input_shape=(height, width, 3))
    inputs = Input(shape=(height, width, 3))

    for layer in vggnet.layers:
        layer.trainable = False
    x = vggnet.layers[1](inputs)  # block1_conv1
    out1 = vggnet.layers[2](x)  # block1_conv2
    x = vggnet.layers[3](out1)  # block1_pool

    x = vggnet.layers[4](x)  # block2_conv1
    out2 = vggnet.layers[5](x)  # block2_conv2
    x = vggnet.layers[6](out2)  # block2_pool

    x = vggnet.layers[7](x)  # block3_conv1
    x = vggnet.layers[8](x)  # block3_conv2
    out3 = vggnet.layers[9](x)  # block3_conv3
    x = vggnet.layers[10](out3)  # block3_pool

    x = vggnet.layers[11](x)  # block3_conv1
    x = vggnet.layers[12](x)  # block3_conv2
    out4 = vggnet.layers[13](x)  # block3_conv3}

    model = Model(inputs=inputs, outputs=[out1, out2, out3, out4])
    return model


# gram_matric(x) is from keras example neural_style_transfer.py
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


img_nrows = 256
img_ncols = 256


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25), axis=[1,2,3])


def featureLoss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1,2,3])


def styleLoss1(y_true, y_pred):
    gm = []
    for i in range(0,1):
        gm.append(K.sum(K.square(gram_matrix(y_true[i,:,:,:])-gram_matrix(y_pred[i,:,:,:])) / ((256*256*64)**2), axis=[0,1]))
    gm = K.stack(gm, axis=0)
    return gm


def styleLoss2(y_true, y_pred):
    gm = []
    for i in range(0,1):
        gm.append(K.sum(K.square(gram_matrix(y_true[i,:,:,:])-gram_matrix(y_pred[i,:,:,:])) / ((128*128*128)**2), axis=[0,1]))
    gm = K.stack(gm, axis=0)
    return gm


def styleLoss3(y_true, y_pred):
    gm = []
    for i in range(0,1):
        gm.append(K.sum(K.square(gram_matrix(y_true[i,:,:,:])-gram_matrix(y_pred[i,:,:,:])) / ((64*64*256)**2), axis=[0,1]))
    gm = K.stack(gm, axis=0)
    return gm


def styleLoss4(y_true, y_pred):
    gm = []
    for i in range(0,1):
        gm.append(K.sum(K.square(gram_matrix(y_true[i,:,:,:])-gram_matrix(y_pred[i,:,:,:])) / ((32*32*512)**2), axis=[0,1]))
    gm = K.stack(gm, axis=0)
    return gm


def TVLoss(y_true, y_pred):
    return total_variation_loss(y_pred)

