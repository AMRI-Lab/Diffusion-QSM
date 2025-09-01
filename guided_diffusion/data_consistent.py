"""
functions for QSM-related data consistency
"""

import torch as th
import numpy as np

from guided_diffusion.dipole_utils import myfftnc, myifftnc

def overlapping_grid_indices(x,output_size,r=None):
    _,_,d,h,w = x.shape
    d_list = [i for i in range(0,d-output_size[0] + 1, r[0])]
    h_list = [i for i in range(0,h-output_size[1] + 1, r[1])]
    w_list = [i for i in range(0,w-output_size[2] + 1, r[2])]
    return d_list, h_list, w_list


def patch_process(patch1,patch2,overlap,direction):
    """
    This function is called to stitch the image patches 
    patch1: patch to be stitched
    patch2: the patch overlapped with patch1
    direction: The direction of stitching
    """       
    size1=patch1.shape
    size2=patch2.shape
    result=patch1
    if size1==size2:
        size=size1
    else:
        raise ValueError("two patches don't have the same size")

    weight=np.ones(overlap) 
    for x in range(int(overlap)):
        weight[x]=1-x/(overlap-1)

    if direction==0:
        block1=patch1[0:overlap,:,:,:]
        block2=patch2[int(size[0]-overlap):size[0],:,:,:]
        for i in range(int(overlap)):
            result[i,:,:,:]=weight[i]*block2[i,:,:,:]+(1-weight[i])*block1[i,:,:,:]
    
    elif direction==1:
        block1=patch1[:,0:overlap,:,:]
        block2=patch2[:,int(size[1]-overlap):size[1],:,:]
        for j in range(int(overlap)):
            result[:,j,:,:]=weight[j]*block2[:,j,:,:]+(1-weight[j])*block1[:,j,:,:]
            
    elif direction==2:
        block1=patch1[:,:,0:overlap,:]
        block2=patch2[:,:,int(size[2]-overlap):size[2],:]
        for k in range(int(overlap)):
            result[:,:,k,:]=weight[k]*block2[:,:,k,:]+(1-weight[k])*block1[:,:,k,:]
    else:
        raise ValueError("direction must be a integer between 0 and 2")

    return result
    

def data_consistency_qsm_ki_gc(img, args1, step):
    '''        
    applying data consistency of the dipole model in image domain
    '''        
    def A(x,m,D,calci):
        return calci*(myifftnc(D * myfftnc(x,[-3,-2,-1]), [-3,-2,-1]))
    def AT(phi,m,D,calci):
        return myifftnc(D * myfftnc(calci*phi,[-3,-2,-1]), [-3,-2,-1])
    def AtA(x,m,D,calci,lam):
        return AT(A(x,m,D,calci),m,D,calci) + lam * x 

    lam = args1['lam'][step]
    lam = th.Tensor(np.array(lam))
    lam = th.complex(lam, th.tensor(0.)).to(img.device)
    

    tissuePhi = args1['measurement_new'].to(img.device)
    D = args1['dipole_kernel'].to(img.device)
    mask = args1['msk'].to(img.device)
    calci = args1['calci'].to(img.device)

    z_k = img 
                        
    x = th.zeros_like(img) 
        
    Aty = AT(tissuePhi,mask,D,calci).real 
        
    rhs = Aty + lam * z_k 

    i, r, p = 0, rhs, rhs
    rTr = th.sum(r.conj()*r).real.to(th.float32)
        
    numbs = args1['lam_step'] 
        
    while i < numbs and th.real(rTr) > 1e-8: 
        Ap = AtA(p,mask,D,calci,lam)
        alpha = rTr / th.sum(p.conj()*Ap).real.to(th.float32)
        alpha = th.complex(alpha, th.tensor(0.).to(img.device))
        x = x + alpha * p 
                        
        r = r - alpha * Ap
        rTrNew = th.sum(r.conj()*r).real.to(th.float32)
            
        beta = rTrNew / rTr
        beta = th.complex(beta, th.tensor(0.).to(img.device))
        p = r + beta * p 
            
        i += 1
        rTr = rTrNew

    out = th.real(x)

    return out
