import argparse
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import optimizers
import imageio
import os
from models import *
from utils.utils import prctile_norm
import tifffile as tiff

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../dataset/test/F-actin/3d")
parser.add_argument("--folder_test", type=str, default="input_raw_sim_images")
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--gpu_memory_fraction", type=float, default=0)
parser.add_argument("--model_name", type=str, default="caGAN3D")
parser.add_argument("--model_weights", type=str, default="../trained_models/3d/caGAN3D-SIM_F-actin/weights_best.h5")
parser.add_argument("--ndirs", type=int, default=3)
parser.add_argument("--nphases", type=int, default=5)

args = parser.parse_args()
gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
data_dir = args.data_dir
folder_test = args.folder_test
model_name = args.model_name
model_weights = args.model_weights
ndirs = args.ndirs
nphases = args.nphases

output_name = 'output_' + model_name + '-'
test_images_path = data_dir + '/' + folder_test
output_dir = data_dir + '/' + output_name

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# --------------------------------------------------------------------------------
#                              glob test data path
# --------------------------------------------------------------------------------
img_path = glob.glob(test_images_path + '/*.tif')
img_path.sort()
n_channel = ndirs * nphases
img = tiff.imread(img_path[0])
shape = img.shape
input_y, input_x = shape[1], shape[2]
input_z = shape[0] // n_channel
output_dir = output_dir + 'SIM'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# --------------------------------------------------------------------------------
#                          select models and load weights
# --------------------------------------------------------------------------------
modelFns = {'caGAN3D': caGAN3D.Generator}
modelFN = modelFns[model_name]
optimizer = optimizers.Adam(lr=1e-5, decay=0.5)
m = modelFN((input_y, input_x, input_z, n_channel))
m.load_weights(model_weights)
m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

print('Processing ' + test_images_path + '...')
im_count = 0
for curp in img_path:

    img = tiff.imread(curp)
    img = np.array(img).reshape((n_channel, input_z, input_y, input_x), order='F').transpose((2, 3, 1, 0))
    img = img[np.newaxis, :]

    img = prctile_norm(img)
    pr = prctile_norm(np.squeeze(m.predict(img)))

    outName = curp.replace(test_images_path, output_dir)

    pr = np.transpose(65535 * pr, (2, 0, 1)).astype('uint16')
    tiff.imwrite(outName, pr, dtype='uint16')




