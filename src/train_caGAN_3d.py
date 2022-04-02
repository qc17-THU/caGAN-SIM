import argparse
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import datetime
import glob
import os
import tensorflow as tf
from models import *
from utils.lr_controller import ReduceLROnPlateau
from utils.data_loader import data_loader_multi_channel_3d, data_loader_multi_channel_3d_wf
from utils.utils import img_comp
from utils.loss import loss_mse_ssim_3d, create_psf_loss
from utils.psf_generator import parameters3D, cal_psf_3d, psf_estimator_3d

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=4)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.3)
parser.add_argument("--mixed_precision_training", type=int, default=1)
# parser.add_argument("--data_dir", type=str, default="../dataset/train/F-actin_3D")
parser.add_argument("--data_dir", type=str, default="/media/zkyd/New Volume1/Enscosin_GFP/ToSIM-3D/TrainData_Cell1-22_SNR-GT_WF_Raw_ToSIM_1.27NA_Test")
parser.add_argument("--save_weights_dir", type=str, default="../trained_models/3d")
parser.add_argument("--model_name", type=str, default="caGAN3D")
parser.add_argument("--patch_y", type=int, default=64)
parser.add_argument("--patch_x", type=int, default=64)
parser.add_argument("--patch_z", type=int, default=11)
parser.add_argument("--input_channels", type=int, default=15)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--norm_flag", type=int, default=1)
parser.add_argument("--iterations", type=int, default=1000000)
parser.add_argument("--sample_interval", type=int, default=2)
parser.add_argument("--validate_interval", type=int, default=5)
parser.add_argument("--validate_num", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--d_start_lr", type=float, default=1e-6)  # 2e-5
parser.add_argument("--g_start_lr", type=float, default=1e-4)  # 1e-4
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--load_weights", type=int, default=1)
parser.add_argument("--optimizer_name", type=str, default="adam")
parser.add_argument("--train_discriminator_times", type=int, default=1)
parser.add_argument("--train_generator_times", type=int, default=3)
parser.add_argument("--weight_wf_loss", type=float, default=0.05)
parser.add_argument("--wave_len", type=int, default=488)

args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
data_dir = args.data_dir
save_weights_dir = args.save_weights_dir
validate_interval = args.validate_interval
batch_size = args.batch_size
d_start_lr = args.d_start_lr
g_start_lr = args.g_start_lr
lr_decay_factor = args.lr_decay_factor
patch_y = args.patch_y
patch_x = args.patch_x
patch_z = args.patch_z
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
validate_num = args.validate_num
iterations = args.iterations
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name
sample_interval = args.sample_interval
train_discriminator_times = args.train_discriminator_times
train_generator_times = args.train_generator_times
weight_wf_loss = args.weight_wf_loss
wave_len = args.wave_len

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

data_name = data_dir.split('/')[-1]
save_weights_name = model_name + '-SIM_' + data_name

train_images_path = data_dir + '/training/'
train_wf_path = data_dir + '/training_wf/'
train_gt_path = data_dir + '/training_gt/'
validate_images_path = data_dir + '/validate/'
validate_gt_path = data_dir + '/validate_gt/'
save_weights_path = save_weights_dir + '/' + save_weights_name + '/'
sample_path = save_weights_path + 'sampled_img/'

if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)


# --------------------------------------------------------------------------------
#                             Read OTF and PSF
# --------------------------------------------------------------------------------
pParam = parameters3D()
OTF_Path = {488: './OTF/3D-488-OTF-smallendian.mrc', 560: './OTF/3D-560-OTF-smallendian.mrc'}
psf, OTF = cal_psf_3d(OTF_Path[wave_len],  pParam.Ny, pParam.Nx, pParam.Nz, pParam.dky, pParam.dkx, pParam.dkz)
sigma_y, sigma_x, sigma_z = psf_estimator_3d(psf)
ksize = int(sigma_y * 4)
halfy = pParam.Ny // 2
psf = psf[halfy-ksize:halfy+ksize, halfy-ksize:halfy+ksize, :]
psf = np.reshape(psf, (2*ksize, 2*ksize, pParam.Nz, 1, 1)).astype(np.float32)

# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'caGAN3D': caGAN3D}
modelFN = modelFns[model_name]
optimizer_d = optimizers.Adam(lr=d_start_lr, beta_1=0.9, beta_2=0.999)
optimizer_g = optimizers.Adam(lr=g_start_lr, beta_1=0.9, beta_2=0.999)

# --------------------------------------------------------------------------------
#                           define discriminator model
# --------------------------------------------------------------------------------
d = modelFN.Discriminator((patch_y * scale_factor, patch_x * scale_factor, patch_z, 1))
d.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])

# --------------------------------------------------------------------------------
#                              define combined model
# --------------------------------------------------------------------------------
frozen_d = Model(inputs=d.inputs, outputs=d.outputs)
frozen_d.trainable = False
g = modelFN.Generator((patch_y, patch_x, patch_z, input_channels))
input_lp = Input((patch_y, patch_x, patch_z, input_channels))
fake_hp = g(input_lp)
judge = frozen_d(fake_hp)
label = np.zeros(batch_size)
if weight_wf_loss > 0:
    combined = Model(input_lp, [judge, fake_hp, fake_hp])
    loss_wf = create_psf_loss(psf)
    combined.compile(loss=['binary_crossentropy', loss_mse_ssim_3d, loss_wf], optimizer=optimizer_g,
                     loss_weights=[0.1, 1, weight_wf_loss])  # 0.1 1
else:
    combined = Model(input_lp, [judge, fake_hp])
    combined.compile(loss=['binary_crossentropy', loss_mse_ssim_3d], optimizer=optimizer_g,
                     loss_weights=[0.1, 1])  # 0.1 1

lr_controller_g = ReduceLROnPlateau(model=combined, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                    cooldown=0, min_lr=g_start_lr * 0.1, verbose=1)
lr_controller_d = ReduceLROnPlateau(model=d, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                    cooldown=0, min_lr=d_start_lr * 0.1, verbose=1)

# --------------------------------------------------------------------------------
#                                 about Tensorboard
# --------------------------------------------------------------------------------
log_path = save_weights_path + 'graph'
if not os.path.exists(log_path):
    os.mkdir(log_path)
writer = tf.summary.create_file_writer(log_path)
train_names = ['Generator_loss', 'Discriminator_loss']
val_names = ['val_MSE', 'val_SSIM', 'val_PSNR', 'val_NRMSE']


def write_log(writer, names, logs, batch_no):
    with writer.as_default():
        tf.summary.scalar(names, logs, step=batch_no)
        writer.flush()


# --------------------------------------------------------------------------------
#                             Sample and validate
# --------------------------------------------------------------------------------
def Validate(iter, sample=0):
    validate_path = glob.glob(validate_images_path + '*')
    validate_path.sort()

    if sample == 1:
        validate_path = np.random.choice(validate_path, size=1)
    elif validate_num < validate_path.__len__():
        validate_path = validate_path[0:validate_num]

    mses, nrmses, psnrs, ssims = [], [], [], []
    imgs, imgs_gt, output = [], [], []
    for path in validate_path:
        [imgs, imgs_gt] = data_loader_multi_channel_3d([path], validate_images_path, validate_gt_path, patch_y,
                                                       patch_x, patch_z, 1, norm_flag=norm_flag)
        output = np.squeeze(g.predict(imgs))
        output_proj = np.max(output, 2)
        gt_proj = np.max(np.squeeze(imgs_gt), 2)
        mses, nrmses, psnrs, ssims = img_comp(gt_proj, output_proj, mses, nrmses, psnrs, ssims)

    if sample == 0:
        # if best, save weights.best
        g.save_weights(save_weights_path + 'weights_latest.h5')
        d.save_weights(save_weights_path + 'weights_disc_latest.h5')
        if min(validate_nrmse) > np.mean(nrmses):
            g.save_weights(save_weights_path + 'weights_best.h5')
            d.save_weights(save_weights_path + 'weights_disc_best.h5')

        validate_nrmse.append(np.mean(nrmses))
        curlr_g = lr_controller_g.on_epoch_end(iter, np.mean(nrmses))
        curlr_d = lr_controller_d.on_epoch_end(iter, np.mean(nrmses))
        write_log(writer, val_names[0], np.mean(mses), iter)
        write_log(writer, val_names[1], np.mean(ssims), iter)
        write_log(writer, val_names[2], np.mean(psnrs), iter)
        write_log(writer, val_names[3], np.mean(nrmses), iter)
        write_log(writer, 'lr_g', curlr_g, iter)
        write_log(writer, 'lr_d', curlr_d, iter)

    else:
        imgs = np.mean(imgs, 4)
        r, c = 3, patch_z
        plt.figure(figsize=(22, 6))
        for j in range(c):
            plt.subplot(r, c, j + 1)
            plt.imshow(imgs[0, :, :, j])
            plt.axis("off")
            plt.subplot(r, c, j + c + 1)
            plt.imshow(output[:, :, j])
            plt.axis("off")
            plt.subplot(r, c, j + 2 * c + 1)
            plt.imshow(imgs_gt[0, :, :, j, 0])
            plt.axis("off")
        plt.savefig(sample_path + '%d.png' % iter)


# --------------------------------------------------------------------------------
#                             if exist, load weights
# --------------------------------------------------------------------------------
if load_weights:
    if os.path.exists(save_weights_path + 'weights_best.h5'):
        g.save_weights(save_weights_path + 'weights_best.h5')
        d.save_weights(save_weights_path + 'weights_disc_best.h5')
        print('Loading weights successfully: ' + save_weights_path + 'weights_best.h5')
    elif os.path.exists(save_weights_path + 'weights_latest.h5'):
        g.save_weights(save_weights_path + 'weights_latest.h5')
        d.save_weights(save_weights_path + 'weights_disc_latest.h5')
        print('Loading weights successfully: ' + save_weights_path + 'weights_latest.h5')


# --------------------------------------------------------------------------------
#                                    training
# --------------------------------------------------------------------------------
# label
batch_size_d = round(batch_size / 2)
valid_d = np.ones(batch_size_d).reshape((batch_size_d, 1))
fake_d = np.zeros(batch_size_d).reshape((batch_size_d, 1))
label = np.concatenate((valid_d, fake_d), axis=0)
valid = np.ones(batch_size).reshape((batch_size, 1))
fake = np.zeros(batch_size).reshape((batch_size, 1))

# initialization
start_time = datetime.datetime.now()
gloss_record = []
dloss_record = []
lr_controller_g.on_train_begin()
lr_controller_d.on_train_begin()
validate_nrmse = [np.Inf]
images_path = glob.glob(train_images_path + '*')
for it in range(iterations):

    # ------------------------------------
    #         train discriminator
    # ------------------------------------
    input_d, gt_d = data_loader_multi_channel_3d(images_path, train_images_path, train_gt_path,
                                                 patch_y, patch_x, patch_z, batch_size_d, norm_flag=norm_flag)
    fake_input_d = g.predict(input_d)
    input_d = np.concatenate((gt_d, fake_input_d), axis=0)
    loss_discriminator = d.train_on_batch(input_d, label)
    dloss_record.append(loss_discriminator[0])
    for i in range(train_discriminator_times - 1):
        input_d, gt_d = data_loader_multi_channel_3d(images_path, train_images_path, train_gt_path,
                                                     patch_y, patch_x, patch_z, batch_size_d, norm_flag=norm_flag)
        fake_input_d = g.predict(input_d)
        input_d = np.concatenate((gt_d, fake_input_d), axis=0)
        loss_discriminator = d.train_on_batch(input_d, label)
        dloss_record.append(loss_discriminator[0])

    # ------------------------------------
    #         train generator
    # ------------------------------------
    if weight_wf_loss > 0:
        input_g, wf_g, gt_g = data_loader_multi_channel_3d_wf(images_path, train_images_path, train_wf_path,
                                                              train_gt_path, patch_y, patch_x, patch_z, batch_size,
                                                              norm_flag=norm_flag)
        loss_generator = combined.train_on_batch(input_g, [valid, gt_g, wf_g])
    else:
        input_g, gt_g = data_loader_multi_channel_3d(images_path, train_images_path, train_gt_path,
                                                     patch_y, patch_x, patch_z, batch_size, norm_flag=norm_flag)
        loss_generator = combined.train_on_batch(input_g, [valid, gt_g])
    gloss_record.append(loss_generator[2])

    for i in range(train_generator_times - 1):
        if weight_wf_loss > 0:
            input_g, wf_g, gt_g = data_loader_multi_channel_3d_wf(images_path, train_images_path, train_wf_path,
                                                                  train_gt_path, patch_y, patch_x, patch_z, batch_size,
                                                                  norm_flag=norm_flag)
            loss_generator = combined.train_on_batch(input_g, [valid, gt_g, wf_g])
        else:
            input_g, gt_g = data_loader_multi_channel_3d(images_path, train_images_path, train_gt_path,
                                                         patch_y, patch_x, patch_z, batch_size, norm_flag=norm_flag)
            loss_generator = combined.train_on_batch(input_g, [valid, gt_g])

        gloss_record.append(loss_generator[2])

    elapsed_time = datetime.datetime.now() - start_time
    print("%d epoch: time: %s, d_loss = %.5s, d_acc = %.5s, g_loss = %s" % (
        it + 1, elapsed_time, loss_discriminator[0], loss_discriminator[1], loss_generator[2]))

    if (it + 1) % sample_interval == 0:
        images_path = glob.glob(train_images_path + '*')
        Validate(it + 1, sample=1)

    if (it + 1) % validate_interval == 0:
        Validate(it + 1, sample=0)
        write_log(writer, train_names[0], np.mean(gloss_record), it + 1)
        write_log(writer, train_names[1], np.mean(dloss_record), it + 1)
        gloss_record = []
        dloss_record = []
