addpath('/scripts')

%%
%folder
cd data 

load data.mat
% the data.mat contains the following: 
% phi: tissue phase (normalized, unit: ppm)
% msk: brain mask

vox=[1,1,2]; % resolution
H=[0,0,1];% scan direction

patch_size=[64,64,64]; % patch size
stride_size=[48,48,48]; % stride size 

%%
mask=msk;
phi_use=phi;

pad_size(1)=estimate_size_for_patch(size(phi_use,1),patch_size(1),stride_size(1));
pad_size(2)=estimate_size_for_patch(size(phi_use,2),patch_size(2),stride_size(2));
pad_size(3)=estimate_size_for_patch(size(phi_use,3),patch_size(3),stride_size(3));

phi_use=padarray(phi_use,[pad_size(1)/2,pad_size(2)/2,pad_size(3)/2],0,'both');
mask=padarray(mask,[pad_size(1)/2,pad_size(2)/2,pad_size(3)/2],0,'both');

D2=calcD2Matrix(size(phi_use),vox,H);

save data_DiffusionQSM.mat phi_use mask D2 pad_size
