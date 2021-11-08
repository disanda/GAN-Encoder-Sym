

# Symmtrical-GAN 

![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-blue.svg?style=plastic) 
![CUDA 10.2](https://img.shields.io/badge/cuda-10.2-blue.svg?style=plastic)
![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=plastic)

>  This is the backup code  for  "Fast Transformation of Discriminators into Encoders using Pre-Trained GANs". 


##  Usage

###  1. Training DCGAN 

> python E_align.py

Tips: please refer to the below parameters to implement our ablation study (change ZoutDim 1 to 128 that equal with Zdim, and Zdim is G input dim).


- case 1 -- G_in: (128,2048) | D_out: (2048, 1):

>ep100-Celeba_HQ-Gscale8-GDscale8-Dscale1-Zdim128-ZoutDim1-Hidden_Scale2-img_size256-batch_size30-BNFalse-GDstdFalse-GreluTrue

- case 2 -- G_in: (128,2048) | D_out: (2048, 2):

>ep100-Celeba_HQ-Gscale8-GDscale8-Dscale1-Zdim128-ZoutDim2-Hidden_Scale2-img_size256-batch_size30-BNFalse-GDstdFalse-GreluTrue

- case 3 -- G_in: (128,2048) | D_out: (2048, 4):

>ep100-Celeba_HQ-Gscale8-GDscale8-Dscale1-Zdim128-ZoutDim4-Hidden_Scale2-img_size256-batch_size30-BNFalse-GDstdFalse-GreluTrue

- case 4--7  ...

- case 8 -- G_in: (128,2048) | D_out: (2048, 128):

>ep100-Celeba_HQ-Gscale8-GDscale8-Dscale1-Zdim128-ZoutDim16-Hidden_Scale2-img_size256-batch_size30-BNFalse-GDstdFalse-GreluTrue


### 2. Training pre-trained PGGAN (D to E)

> python train_PGGAN.py

- Tips: 

> In this case, we can resue the weights of pre-trained D, and transform D to E.

> before training, please download pre-trained models (D and G) to ./checkpoint 

> We also implement other type of PGGAN-FC in train_PGGAN_FC.py.


## Metric and  Pre-trained Models 

- FID

  We use official code of FID and its default setting to evluate our results.

  The FID code is here: https://github.com/mseitzer/pytorch-fid.git

- DataSet

   We can directly download CelebA-HQ in here: https://github.com/switchablenorms/CelebAMask-HQ

  There 30,000 real aligned-face images with 1024x1024 (10,000 for PGGAN evaluation), and we resize image to 256x256 for DCGAN training and evaluation.

- Pre-trained Models

   We offered Pre-trianed model for PGGAN reusing model weights and training here: [google drive](https://drive.google.com/drive/folders/1cBfBgknTuZxLAfICwo8kBEGYNOxJ-hQd?usp=sharing). 

   This link also include DCGAN pre-trained model (Doutput=128).

   If you need more pre-trained models (e.g. PGGAN-FC), please find blow.


##  Baseline (Acknowledgements)

- PGGAN in Pytorch: https://github.com/akanimax/pro_gan_pytorch
- PGGAN_FC: https://github.com/genforce/genforce.git
- DCGAN: https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
