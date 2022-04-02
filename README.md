# caGAN-SIM

**caGAN-SIM software** is a tensorflow implementation for deep learning-based 3D-SIM reconstruction. This repository is developed based on the 2021 IEEE JSTQE paper [**3D Structured Illumination Microscopy via Channel Attention Generative Adversarial Network**](https://doi.org/10.1109/JSTQE.2021.3060762).<br>

Author: Chang Qiao<sup>1,#</sup>, Xingye Chen<sup>1,#</sup>, Siwei Zhang<sup>2,#</sup>, Di Li<sup>2,3</sup>, Yuting Guo<sup>2,3</sup>, Qionghai Dai<sup>1,+</sup>, Dong Li<sup>2,3,4,+</sup><br>
<sup>1</sup>Department of Automation, Tsinghua University, Beijing, China.<br>
<sup>2</sup>National Laboratory of Biomacromolecules, CAS Center for Excellence in Biomacromolecules, Institute of Biophysics, Chinese Academy of Sciences, Beijing, China.<br>
<sup>3</sup>College of Life Sciences, University of Chinese Academy of Sciences, Beijing, China.<br>
<sup>4</sup>Bioland Laboratory, Guangzhou Regenerative Medicine and Health Guangdong Laboratory, Guangzhou, China.<br>
<sup>#</sup>Equal contribution.  
<sup>+</sup>Correspondence to: qhdai@tsinghua.edu.cn and lidong@ibp.ac.cn

## Contents
- [Environment](#environment)
- [File structure](#file-structure)
- [Test pre-trained models](#test-pre-trained-models)
- [Train a new model](#train-a-new-model)
- [License](#License)
- [Citation](#citation)

## Environment
- Ubuntu 16.04
- CUDA 11.0.207
- cudnn 8.0.4
- Python 3.6.10
- Tensorflow 2.4.0
- GPU: GeForce RTX 2080Ti

## File structure
- `./dataset` is the default path for training data and testing data
    - `./dataset/train` The augmented training image patch pairs will be saved here by default
    - `./dataset/test` includes some demo images of F-actin and microtubules to test caGAN-SIM models
- `./src` includes the source codes of caGAN-SIM
	- `./src/models` includes declaration of caGAN models
	- `./src/utils` is the tool package of caGAN-SIM software
- `./trained_models` place pre-trained caGAN-SIM models here for testing, and newly trained models will be saved here by default

## Test pre-trained models
- Place your testing data in `./dataset/test`
- Open your terminal and cd to `./src`
- Run `bash demo_predict.sh` in your terminal. Note that before running the bash file, you should check if the data paths and other arguments in `demo_predict.sh` are set correctly
- The output reconstructed SR images will be saved in `--data_dir`

## Train a new model
- Data for training: You can train a new caGAN-SIM model using your own datasets. Note that you'd better divide the dataset of each specimen into training part and validation/testing part before training, so that you can test your model with the preserved validation/testing data
- Run `bash demo_train.sh` in your terminal to train a new caGAN-SIM model. Similar to testing, before running the bash file, you should check if the data paths and the arguments are set correctly
- You can run `tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph` to monitor the training process via tensorboard. If the validation loss isn't likely to decay any more, you can use early stop strategy to end the training
- Model weights will be saved in `./trained_models/` by default

## License
This repository is released under the MIT License (refer to the LICENSE file for details).

## Citation
If you find the code helpful in your resarch, please cite the following paper:
```
@article{qiao20213d,
  title={3D Structured Illumination Microscopy via Channel Attention Generative Adversarial Network},
  author={Qiao, Chang and Chen, Xingye and Zhang, Siwei and Li, Di and Guo, Yuting and Dai, Qionghai and Li, Dong},
  journal={IEEE Journal of Selected Topics in Quantum Electronics},
  volume={27},
  number={4},
  pages={1--11},
  year={2021},
  publisher={IEEE}
}
```