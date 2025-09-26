# Diffusion-QSM
This repo contains Matlab codes for generating input data and python codes for running Diffusion-QSM.  

Diffusion-QSM: diffusion model with time-travel and resampling refinement for quantitative susceptibility mapping
https://ieeexplore.ieee.org/document/11147123

## Setup   
Our codes are built upon [openai/guided-diffusion](https://github.com/openai/guided-diffusion), you may install the environment based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
* STISuite 3.0
* PyTorch 2.4.1
* Python 3.8

## Introduction to files
1.  `data_prep`

test data preparation, run `demo.m` to generate `data_DiffusionQSM.mat` under `data_prep/data` folder 

2.   `guided_diffusion`

files for DDPM and QSM-related codes, important files include `gaussian_diffusion.py` and `data_consistent.py` 

3.  `weights`

trained weight `model700000.pt` from [Google Drive](https://drive.google.com/file/d/1BZqL7dPCUcRbaik2ygvURg39CGwFsnbE/view?usp=drive_link) 

4.  `QSM_Diffusion_train.py`

training code, train the diffusion network using this script

5.  `recon.py`

inference code, recommend to adjust only in variable `args_params`  




## Usage
**Data preparation**
1.  Generate test data `data_DiffusionQSM.mat` containing `phi_use`, `mask`, `D2` and `pad_size` based on the `demo.m` files in `data_prep` folder. You can download one test data via [Google Drive](https://drive.google.com/file/d/1B0pkNPCZTDhohy7rW6lL72vbq_Rc1vUH/view?usp=sharing). 
2.  Download the trained weight of the network via [Google Drive](https://drive.google.com/file/d/1BZqL7dPCUcRbaik2ygvURg39CGwFsnbE/view?usp=drive_link) and place it in `weights` folder 
3.  Adjust the `args_params` in `recon.py`

**Training**
1.  Generate your own training patch-wise dataset (64x64x64 dimension)
2.  Run `QSM_Diffusion_train.py` for network training

**Testing**
1.  Run `recon.py` for generating Diffusion-QSM output
2.  The results are saved under `data_prep/data/results`

## Contact
Feel free to contact `zhangming430424@gmail.com` for questions/discussions/suggestions.

