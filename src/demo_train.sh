#!/usr/bin/python3.6

# ------------------------------- arguments for 3D model -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --mixed_precision_training: whether use mixed precision training or not
# --data_dir: the root path of training data folder
# --train_input_folder: the input images of network (raw SIM image)
# --train_gt_folder: the gt images of training (GT-SIM image)
# --validate_input_folder: the input images of network (raw SIM image) for validation
# --validate_gt_folder: the gt images for validation (GT-SIM image)
# --model_name: 'caGAN3D' or your own model
# --save_weights_dir: root directory where models weights will be saved in
# --save_weights_name: folder name where model weights will be saved in
# --patch_y: the height of input volume patches
# --patch_x: the width of input volume patches
# --patch_z: number of the slices per input volume patch
# --input_channels: 15 for 3D SIM reconstruction
# --scale_factor: upsampling times of the output
# --iterations: total training iterations
# --batch_size: batch size for training
# --start_lr: initial learning rate of training, typically set as 10-4
# --lr_decay_factor: learning rate decay factor, typically set as 0.5

# --d_start_lr: initial learning rate for the discriminator
# --g_start_lr: initial learning rate for the generator
# --train_discriminator_times: times of training discriminator in each iteration
# --train_generator_times: times of training generator in each iteration

# --weight_wf_loss: scalar weight of wide-field loss, defaut=0.05, 0 for no WF-loss usage
# --wave_len: wave length of excitation light, defining the OTF used for WF-loss calculation

# ------------------------------- train a 3D caGAN model -------------------------------
python train_caGAN_3d.py --gpu_id '4' --gpu_memory_fraction 0.3 --mixed_precision_training 1 \
                         --data_dir "../dataset/train/F-actin_3D" \
                         --save_weights_dir "../trained_models/3d" \
                         --patch_y 64 --patch_x 64 --patch_z 11 --input_channels 15 \
                         --scale_factor 2 --iterations 200000 --sample_interval 200 \
                         --validate_interval 500 --validate_num 200 --batch_size 2 \
                         --d_start_lr 1e-6 --g_start_lr 1e-4 --lr_decay_factor 0.5 \
                         --train_discriminator_times 1 --train_generator_times 3 \
                         --load_weights 0 --model_name "caGAN3D" --optimizer_name "adam" \
                         --weight_wf_loss 0.05 --wave_len 488