import tensorflow as tf
from tensorflow.keras import backend as K


def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss


def loss_mse_ssim_3d(y_true, y_pred):
    ssim_para = 1e-1 # 1e-2
    mse_para = 1

    # SSIM loss and MSE loss
    x = K.permute_dimensions(y_true, (0, 4, 1, 2, 3))
    y = K.permute_dimensions(y_pred, (0, 4, 1, 2, 3))
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss


def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    mae_loss = mae_para * K.mean(K.abs(x-y))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mae_loss + mse_loss


def create_psf_loss(psf):
    def loss_wf(y_true, y_pred):
        # Wide field loss
        x_wf = K.conv3d(y_pred, psf, padding='same')
        x_wf = K.pool3d(x_wf, pool_size=(2, 2, 1), strides=(2, 2, 1), pool_mode="avg")
        x_min = K.min(x_wf)
        x_wf = (x_wf - x_min) / (K.max(x_wf) - x_min)
        wf_loss = K.mean(K.square(y_true - x_wf))
        return wf_loss
    return loss_wf