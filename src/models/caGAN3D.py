
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, add, multiply
from tensorflow.keras.layers import Conv3D, UpSampling3D, LeakyReLU, Lambda
from .common import global_average_pooling3d, conv_block3d


def CALayer(input, channel, reduction=16):
    W = Lambda(global_average_pooling3d)(input)
    W = Conv3D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv3D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def RCAB(input, channel):
    conv = Conv3D(channel, kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(channel, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, n_RCAB):
    conv = input
    for _ in range(n_RCAB):
        conv = RCAB(conv, channel)
    return conv


def Generator(input_shape, channel=64, n_ResGroup=3, n_RCAB=5):

    inputs = Input(input_shape)
    conv = Conv3D(channel, kernel_size=3, padding='same')(inputs)
    n_ResGroup = n_ResGroup
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, channel=channel, n_RCAB=n_RCAB)

    up = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(channel, kernel_size=3, padding='same')(up)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model


def ConvolutionalBlock(input, channel_size):
    conv = Conv3D(channel_size[0], kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.1)(conv)
    conv = Conv3D(channel_size[1], kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.1)(conv)
    return conv


def Discriminator(input_shape):
    inputs = Input(input_shape)
    x0 = Conv3D(32, kernel_size=3, padding='same')(inputs)
    x0 = LeakyReLU(alpha=0.1)(x0)

    x1 = conv_block3d(x0, (32, 64))
    x2 = conv_block3d(x1, (128, 256))
    x3 = Lambda(global_average_pooling3d)(x2)

    y0 = Flatten(input_shape=(1, 1))(x3)
    y1 = Dense(128)(y0)
    y1 = LeakyReLU(alpha=0.1)(y1)
    output = Dense(1, activation='sigmoid')(y1)
    model = Model(inputs=inputs, outputs=output)
    return model