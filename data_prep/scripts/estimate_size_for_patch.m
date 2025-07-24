function output=estimate_size_for_patch(x,patch_size,stride_size)
% x: input origional size
% patch size:
% stride size:

% output: the number of padding size

output=ceil((x-patch_size)/stride_size)*stride_size + patch_size -x   ;

if x<patch_size
    output=ceil((patch_size-x)/1);
end


