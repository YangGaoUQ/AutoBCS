# AutoBCS: Block-based Image Compressive Sensing with Data-driven Acquisition and Non-iterative Reconstruction
This reposiotry is for AutoBCS framwork introduced in the following paper: https://arxiv.org/abs/2009.14706.  This code was built and tested on Centos 7.0 with Nvdia Tesla V100 and Windows 10 environment (python 3.7, pytorch > 1.1)  with GTX 1060. 

# Introduction

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
*** 
### requirements: 
### Pytorch 1.1 or later

to be upadted soon...
________________________________
## Whole Framework of AutoBCS
________________________________
![Whole Framework](https://github.com/YangGaoUQ/AutoBCS/blob/master/img/Fig1.png)
Fig. 1: Schematic representation of our proposed AutoBCS architecture. AutoBCS replaces the traditional BCS approach with a unified image acquisition and reconstruction framework.
________________________________
## Training Data Flow
________________________________
![Network Flow](https://github.com/YangGaoUQ/AutoBCS/blob/master/img/Fig2.png)
