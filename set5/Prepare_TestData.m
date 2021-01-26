%% prepare Test Data.

addpath(genpath('../utilities'))

imds = imageDatastore('./', 'FileExtensions', {'.jpg', '.tif', '.png', '.bmp'},...
        'ReadFcn', @(x) imread(x));

for i = 1 : 1 : length(imds.Files)
    st = num2str(i);
    fname = imds.Files{i}; 
    temp = imread(fname);
    
    temp2 = rgb2ycbcr(temp);
    temp2 = temp2(:,:,1);
    temp2 = im2single(temp2(:,:,1));
    
    temp2 = modcrop(temp2, 32);
    
    [l1, l2] = size(temp2);
    if l1 > l2
        temp2 = temp2'; 
    end
    img_label = temp2;
    
    %% 
    fname2 = strcat('(',st,').mat');
    save(fname2, 'img_label')
end 
