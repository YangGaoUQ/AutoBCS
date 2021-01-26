
%%% Generate the training data.

clear;close all;

addpath(genpath('../utilities'))

batchSize      = 64;        %%% batch size
max_numPatches = batchSize*1400; 
% modelName      = 'model_64_96_Adam';
% sigma          = 25;         %%% Gaussian noise level

%%% training and testing
folder_train  = 'BSDS500/data/images/train';  %%% training
folder_test   = 'BSDS500/data/images/test';%%% testing
size_input    = 96;          %%% training
size_label    = 96;          %%% testing
stride_train  = 57;          %%% training
stride_test   = 57;          %%% testing
val_train     = 0;           %%% training % default
val_test      = 1;           %%% testing  % default

%%% training patches
[inputs, labels, set]  = patches_generation(size_input,size_label,stride_train,folder_train,val_train,max_numPatches,batchSize);
%%% testing  patches
[inputs2,labels2,set2] = patches_generation(size_input,size_label,stride_test,folder_test,val_test,max_numPatches,batchSize);

inputs   = cat(4,inputs,inputs2);      clear inputs2;
labels   = cat(4,labels,labels2);      clear labels2;
set      = cat(2,set,set2);            clear set2;

if ~exist('../trainPatch', 'dir')
    mkdir ../trainPatch
end

for i = 1 : size(labels,4)
    subim_input = squeeze(labels(:,:,:,i));
    fname = strcat('./trainPatch/(',num2str(i), ').mat');
    save(fname, 'subim_input')
end

path = '../Training/test_IDs.txt';
fid = fopen(path,'w');
for i = 1: size(labels,4)
    fprintf(fid,'%s \n',num2str(i));
end

fclose(fid);


