"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.data_consistent import overlapping_grid_indices
from guided_diffusion.scheduler import get_schedule_jump, space_timesteps

import random
import nibabel as nib
from scipy.io import loadmat

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args0,
         args1):

    if args1['seed_flag']:
        setup_seed(args1['seed_id'])

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args0, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args0.model_path, map_location=th.device('cuda:{}'.format(str(args1['gpu']) if th.cuda.is_available() else 'cpu'  )) ,
                                  weights_only=True)
    )
    logger.log("loaded...")
    model.to(dist_util.dev_sample(args1['gpu']))
    if args0.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    
    test_Flag = True
    if test_Flag:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop_patch if not args0.use_ddim else diffusion.ddim_sample_loop_patch
        )
        data_size = args1['measurement'].shape
        #########################################################################
        sample = sample_fn(
            model,
            (1, 1, data_size[-3],data_size[-2],data_size[-1]),
            args1,
            clip_denoised=args0.clip_denoised,
            model_kwargs=model_kwargs,
        )
        mask = args1['msk'].squeeze(0).squeeze(0).numpy()
        voxel_size = args1['vox']
        pad_size = args1['padsize']
        affine = np.array([[voxel_size[0], 0, 0, 0], [0, voxel_size[1], 0, 0], [0, 0, voxel_size[2], 0], [0, 0, 0, 1], ])
        sample = sample[0]
        sample = sample.contiguous()
        sample = sample[0]
        sample = sample.cpu().numpy()
        sample = sample * mask
        sample = sample[int(pad_size[0]/2):data_size[-3]-int(pad_size[0]/2),
                        int(pad_size[1]/2):data_size[-2]-int(pad_size[1]/2),
                        int(pad_size[2]/2):data_size[-1]-int(pad_size[2]/2)]
        nib.save(nib.Nifti1Image( sample, affine), args1['tmp_save_path']+'.nii.gz')
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(clip_denoised=True,
                    num_samples=1,
                    batch_size=1,
                    model_path=''
                    )
    defaults.update(model_and_diffusion_defaults())
    defaults['use_ddim'] = True
    defaults['timestep_respacing'] = ''
    defaults['predict_xstart'] = False
    defaults['learn_sigma'] = True
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def create_dic_data(data_root_name):
    data_name='data_DiffusionQSM.mat'
    p_size=[64,64,64]
    stride=[48,48,48]
    
    data1=loadmat(data_root_name+'/'+data_name)
    phi=data1['phi_use']
    dipole_kernel=data1['D2']
    mask=data1['mask']
    calci=data1['calci_use'] if 'calci_use' in data1 else np.ones_like(phi)
    padsize=[data1['pad_size'][0][0],data1['pad_size'][0][1],data1['pad_size'][0][2]]
    
    phi=th.Tensor(phi.copy()).unsqueeze(0).unsqueeze(0)
    
    dipole_kernel=th.Tensor(dipole_kernel.copy()).unsqueeze(0).unsqueeze(0)
    msk__=th.Tensor(mask.copy()).unsqueeze(0).unsqueeze(0)
    calci=th.Tensor(calci.copy()).unsqueeze(0).unsqueeze(0)
    
    d_list,h_list,w_list=overlapping_grid_indices(phi,p_size,r=stride)
    corners=[(i,j,k) for k in w_list for j in h_list for i in d_list]
    x_grid_mask=th.zeros_like(phi)
    for (di,hi,wi) in corners:
        x_grid_mask[:,:,di:di+p_size[0],hi:hi+p_size[1],wi:wi+p_size[2]] += 1
    assert x_grid_mask.min().item()>0
    
    #########################################################################
    args1 = dict(
            save_dir=data_root_name,
            measurement=phi,
            calci=calci,
            dipole_kernel=dipole_kernel,
            msk=msk__,            
            vox=[1,1,1],
            temp_save_flag=True,
            seed_flag=True,
            padsize=padsize, 
            p_size=p_size,
            stride=stride,
            corners=corners,
            x_grid_mask=x_grid_mask,
            weight_mat=None,
            manual_batching=False
    )
    
    return args1


if __name__ == "__main__":
    
    #change the following 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_name', type=str, default='./data_prep/data/',
                        help='path of the test data')
    parser.add_argument('--vox', type=float, nargs=3, default=[1,1,2], help='image resolution')
    
    parser.add_argument('--network_root_name', type=str, default='./weights/model700000.pt',
                        help='trained weight of the diffusion network')
    
    parser.add_argument('--fast_timesteps', type=int, default=200, help='time steps for regular sampling')
    parser.add_argument('--starting_point', type=int, default=10, help='L value for time-travel and resampling, default setting')
    parser.add_argument('--repetition_Time', type=int, default=2, help='r value for time-travel and resampling, default setting')
    
    parser.add_argument('--lam', type=float, default=0.1, help='lambda weighting in CG algorithm')
    parser.add_argument('--CG_iter', type=float, default=15, help='maximum iteration in CG algorithm')
    
    parser.add_argument('--gpu', type=int, default=3, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='seed id')

    args_params = parser.parse_args()
        
    
    
    
    
    
    
    
    
    #don't change the following 
    #####################################
    #####################################
    #####################################
    args0 = create_argparser().parse_args()
    args0.model_path = args_params.network_root_name
    fast_timesteps = args_params.fast_timesteps
    args1 = create_dic_data(args_params.data_root_name)
    
    section_counts = str(fast_timesteps)
    a1 = space_timesteps(1000, section_counts)
    args0.timestep_respacing = ['add',a1]

    args1['gpu'] = args_params.gpu
    args1['vox'] = args_params.vox
    args1['seed_id'] = args_params.seed
    
    args1['lam'] = list(np.linspace(args_params.lam,args_params.lam, 1000 ))
    args1['lam_step'] = args_params.CG_iter
    args1['temp_save_flag'] = True
    
    args1['t_T'] = fast_timesteps
                 
    ss4 = args_params.starting_point  
    ss3 = args_params.repetition_Time+1 #old code, +1 is needed
         
    t_T2 = ss4
    times1 = get_schedule_jump(args1['t_T'],1,1,1)
    times1 = times1[:-(t_T2+1)]
    
    times2 = get_schedule_jump(t_T2+1,1,1,int(ss3))

    times = times1+times2
    time_pairs = list(zip(times[:-1], times[1:]))
                                          
    for ii in range(len(time_pairs)):
        if time_pairs[ii][0] == time_pairs[ii][1]:
            time_pairs.pop(ii)
            break
                           
    all_steps = 0
    for ii in range(len(time_pairs)):
        if time_pairs[ii][0] > time_pairs[ii][1]:
            all_steps = all_steps+1
    print('all timesteps are ' + str(all_steps))
    
    args1['time_pairs'] = time_pairs
                                          
    args1['save_prefix1'] = 'results'
    
    if not os.path.exists(args_params.data_root_name+'/'+args1['save_prefix1']):
        os.mkdir(args_params.data_root_name+'/'+args1['save_prefix1'])
    args1['save_prefix2'] = 'sample_DiffusionQSM'
    args1['tmp_save_path'] = args_params.data_root_name+'/'+args1['save_prefix1']+'/'+args1['save_prefix2']
    
    main(args0, args1)
                
                