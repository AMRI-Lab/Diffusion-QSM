#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:59:28 2023

@author: zhangm
"""
import torch
from torch import fft
import numpy as np

def myfftnc(x, dim):
    #device = x.device
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.ifftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)

def myifftnc(x, dim):
    #device = x.device
    if dim is None:
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)

