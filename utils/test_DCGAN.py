# this code could evaluate DCGAN to Symmetrical GAN
import functools
import numpy as np
import tensorboardX
import torch
import tqdm
import argparse
import os
import yaml
import torchvision
import utils.data_tools as data
import networks.DCGAN as net #通过参数Gscale 和 Dscale4g 控制 G和D参数规模的网络
import utils.loss_func as loss_func
from torchsummary import summary
import itertools
import lpips
from utils.utils import set_seed

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--iterations', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=1) # STL:100
parser.add_argument('--experiment_name', default=None)
parser.add_argument('--img_size',type=int, default=256) # STL:128, CelebA-HQ: 256
parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
parser.add_argument('--dataname', default='Celeba_HQ') #choices=['mnist','fashion_mnist','cifar10', 'STL10',  'celeba','Celeba_HQ'] and so on.
parser.add_argument('--datapath', default='./dataset/CelebA-HQ-img/') 
parser.add_argument('--data_flag', type=bool, default=False) # True: STL10, False: Celeba_HQ
parser.add_argument('--z_dim', type=int, default=128) # input to G
parser.add_argument('--z_out_dim', type=int, default=1) # output from D
parser.add_argument('--Gscale', type=int, default=8) # G parameter size, scale：网络隐藏层维度数,默认为 image_size//8 * image_size 
parser.add_argument('--GDscale', type=int, default=8) # D parameter size (ratio with G),Gscale的规模(在D中默认和Gscale相同)
parser.add_argument('--Dscale', type=int, default=1) # Dscale相对Gscale缩小的倍数
parser.add_argument('--BN', type=bool, default=False) # default is SN
parser.add_argument('--hidden_scale', type=int, default=2)
parser.add_argument('--GDstd', type=bool, default=False) # GD的训练损失加std()约束
parser.add_argument('--Grelu', type=bool, default=True) # 默认False是Leaky-Relu
args = parser.parse_args()

#Fashion_mnist:  img_size=32, z_dim=32, Gscale=8


# output_dir
if args.experiment_name == None:
    args.experiment_name = 'ep%d_z_out_dim%d_iter%d'%(args.epochs,args.z_out_dim,args.iterations)

if not os.path.exists('output'):
    os.mkdir('output')

output_dir = os.path.join('output', args.experiment_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


sample_dir = os.path.join(output_dir, 'samples')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


# ==============================================================================
# =                                    samples                                     =
# ==============================================================================

G_Path = './output/ab8-ep100-Celeba_HQ-Gscale8-GDscale8-Dscale1-Zdim128-ZoutDim128-Hidden_Scale2-img_size256-batch_size30-BNFalse-GDstdFalse-GreluTrue/'
G = net.G(input_dim=args.z_dim, output_dim=args.img_channels, image_size=args.img_size, Gscale=args.Gscale, hidden_scale=args.hidden_scale, BN = args.BN, relu = args.Grelu ).to(device)
G.load_state_dict(torch.load(G_Path,map_location=device))

if __name__ == '__main__':
    for i in tqdm.trange(args.iterations, desc='Epoch Loop'):
        set_seed(i+30000)
        z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
        with torch.no_grad():
            x_sample = G(z)
        torchvision.utils.save_image(x_sample*0.5+0.5,sample_dir+'/%s.jpg'%str(i).rjust(5,'0'))
        print(i)
