# AutoBCS: Block-based Image Compressive Sensing with Data-driven Acquisition and Non-iterative Reconstruction
This reposiotry is for AutoBCS framwork introduced in the following paper: https://arxiv.org/abs/2009.14706.  

This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and Windows 10 environment (python 3.7, pytorch > 1.1)  with GTX 1060. 

# Overview

Abstract—Block compressive sensing is a well-known signal
acquisition and reconstruction paradigm with widespread application
prospects in science, engineering and cybernetic systems.
However, state-of-the-art block-based image compressive sensing
(BCS) methods generally suffer from two issues. The sparsifying
domain and the sensing matrices widely used for image acquisition
are not data-driven, and thus both the features of the image
and the relationships among subblock images are ignored. Moreover,
doing so requires addressing high-dimensional optimization
problems with extensive computational complexity for image
reconstruction. In this paper, we provide a deep learning strategy
for BCS, called AutoBCS, which takes the prior knowledge
of images into account in the acquisition step and establishes
a subsequent reconstruction model for performing fast image
reconstruction with a low computational cost. More precisely,
we present a learning-based sensing matrix (LSM) derived from
training data to accomplish image acquisition, thereby capturing
and preserving more image characteristics than those captured by
existing methods. In particular, the generated LSM is proven to
satisfy the theoretical requirements of compressive sensing, such
as the so-called restricted isometry property. Additionally, we
build a noniterative reconstruction network, which provides an
end-to-end BCS reconstruction framework to eliminate blocking
artifacts and maximize image reconstruction accuracy, in our
AutoBCS architecture. Furthermore, we investigate comprehensive
comparison studies with both traditional BCS approaches
and newly developed deep learning methods. Compared with
these approaches, our AutoBCS framework can not only provide
superior performance in terms of image quality metrics (SSIM
and PSNR) and visual perception, but also automatically benefit
reconstruction speed.

### (1) Whole Framework of AutoBCS

![Whole Framework](https://github.com/YangGaoUQ/AutoBCS/blob/master/img/Fig1.png)
Fig. 1: Schematic representation of our proposed AutoBCS architecture. AutoBCS replaces the traditional BCS approach with a unified image acquisition and reconstruction framework.  

### (2) Training Data Flow

![Network Flow](https://github.com/YangGaoUQ/AutoBCS/blob/master/img/Fig2.png)
Fig. 2: The deep neural network architecture of AutoBCS contains two components: a data-driven image acquisition module
and a noniterative data reconstruction module (composed of an initial reconstruction subnetwork and an octave reconstruction
subnetwork).

## Requirements
Python 3.7 or later  
NVDIA GPU (CUDA 10.0)  
Pytorch 1.10 or later  
MATLAB 2017b or later  

# Manual

## Quick Test (inference on Set 5)
1. Clone this repository

```
    git clone https://github.com/YangGaoUQ/AutoBCS.git
```

2. Run the following scripts (in Folder './Inference/') to test the pre-trained models.

```python
    python Evaluate_set5.py
```

## The whole test pipeline (on your own data)
1. Prepare your test data, and make your own directory for it, and rename them in a numerical order. (You can use Prepare_TestData.m provided in the folder './set5/' to process your data.) 
```matlab 
    matlab -r "Prepare_TestData.m"
```

2. Modify the  test code. 
    1. Open ./Inference/Evaluate_set5.py using your own IDE
    2. go to line 37, set File_No = numer_of_your_own_images
    3. go to line 38, change 'set5' to your own directory
    4. save it as your own test script file. 

3. Run your own code

```python
    python your_own_test_script.py  
```
## Train new AutoBCS Net
1. prepare your own trianing datasets (We used BSD500 database https://github.com/BIDS/BSDS500 )

2. Preprocessing data sets using the codes in the directory './Preprocessing_for_training' with Matlab
```matlab 
    matlab -r "GenerateData_model_64_96_Adam.m"
```

3. Enter the tranining folder ('./Training/'), and run the code: 
```python 
    python TrainAutoBCS.py 
```



