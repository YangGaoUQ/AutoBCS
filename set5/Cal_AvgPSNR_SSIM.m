%% cal psnr and ssim
clear
clc
spsnr = 0;
sssim = 0;

addpath(genpath('../utilities'))
for i = 1 : 1 : 5
    st = num2str(i);
    fname = strcat('./(',st,').mat');
    fname2 = strcat('./(',st,')_finalRecon.mat');
    load(fname);
    load(fname2);
   
    [temp_psnr, temp_ssim] = Cal_PSNRSSIM(im2uint8(PRED_FR), im2uint8(img_label),0,0);
    spsnr = temp_psnr + spsnr;
    sssim = temp_ssim + sssim;
end 

avg_psnr = spsnr / 5
avg_ssim = sssim / 5
