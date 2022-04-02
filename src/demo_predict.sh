#!/usr/bin/python3.6
# ------------------------------- arguments for 3D model -------------------------------
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --data_dir: the root path of test data folder
# --folder_test: the raw SIM image or WF iamge folder
# --model_name: select 'DFCAN' or 'DFGAN' to perform SR image prediction
# --model_weights: the pre-trained model file to be loaded

# ------------------------------- predict with caGAN-SIM reconstruction model  -------------------------------
python predict_3d.py --gpu_id '4' --gpu_memory_fraction 0.3 \
                     --data_dir "../dataset/test/F-actin/3d" \
                     --folder_test "input_raw_sim_images" \
                     --model_name "caGAN3D" \
                     --model_weights "../trained_models/3d/caGAN3D-SIM_F-actin/weights_best.h5" \
