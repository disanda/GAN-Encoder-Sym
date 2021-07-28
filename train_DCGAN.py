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
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=30) #STL:100
parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--experiment_name', default=None)
parser.add_argument('--img_size',type=int, default=256) # STL:128, CelebA-HQ: 256
parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
parser.add_argument('--dataname', default='Celeba_HQ') #choices=['mnist','cifar10', 'STL10',  'celeba','Celeba_HQ'] and so on.
parser.add_argument('--datapath', default='./dataset/CelebAmask-HQ/CelebA-HQ-img/') 
parser.add_argument('--data_flag', type=bool, default=False) # True: STL10, False: Celeba_HQ
parser.add_argument('--z_dim', type=int, default=128) # input to G
parser.add_argument('--z_out_dim', type=int, default=1) # output from D
parser.add_argument('--Gscale', type=int, default=16) # scale：网络隐藏层维度数,默认为 image_size//8 * image_size 
parser.add_argument('--Dscale', type=int, default=1) 
parser.add_argument('--BN', type=bool, default=False) 
parser.add_argument('--hidden_scale', type=int, default=2) 
args = parser.parse_args()

# output_dir

if args.experiment_name == None:
    args.experiment_name = '%s-Gscale%d-Dscale%d-Zdim%d-ZoutDim%d-Hidden_Scale%d-batch_size%d'%(args.dataname,args.Gscale,args.Dscale,args.z_dim,args.z_out_dim,args.hidden_scale,args.image_size)

if not os.path.exists('output'):
    os.mkdir('output')

output_dir = os.path.join('output', args.experiment_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

ckpt_dir = os.path.join(output_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

sample_dir = os.path.join(output_dir, 'samples_training')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# save settings
with open(os.path.join(output_dir, 'settings.yml'), "w", encoding="utf-8") as f:
    yaml.dump(args, f)


# GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# dataset
data_loader, shape = data.make_dataset(args.dataname, args.batch_size, args.img_size, args.datapath, pin_memory=use_gpu, num_workers=0)
#n_G_upsamplings = n_D_downsamplings = 5 # 3: 32x32  4:64:64 5:128 6:256
print('data-size:    '+str(shape))

# ==============================================================================
# =                                   model                                    =
# ==============================================================================

G = net.G(input_dim=args.z_dim, output_dim=args.img_channels, image_size=args.img_size, Gscale=args.Gscale, hidden_scale=args.hidden_scale, BN = args.BN ).to(device)
D = net.D(output_dim=args.z_out_dim, input_dim=args.img_channels, image_size=args.img_size, Gscale=args.Gscale, Dscale4G=args.Dscale, hidden_scale=args.hidden_scale  ).to(device)
summary(G,(args.z_dim, args.z_out_dim, args.z_out_dim))
summary(D,(args.img_channels, args.img_size, args.img_size))
x,y = net.get_parameter_number(G),net.get_parameter_number(D)
x_GB, y_GB = net.get_para_GByte(x),net.get_para_GByte(y)

with open(output_dir+'/net.txt','w+') as f:
    print(G,file=f)
    print(D,file=f)
    print('-------------------',file=f)
    print(x,file=f)
    print(x_GB,file=f)
    print(y,file=f)
    print(y_GB,file=f)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = loss_func.get_adversarial_losses_fn(args.adversarial_loss_mode)


# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

mse_loss = torch.nn.MSELoss()

# ==============================================================================
# =                                    Train                                     =
# ==============================================================================

if __name__ == '__main__':

    # main loop
    writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))

    seed_flag = 0

    G.train()
    D.train()
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        it_d, it_g = 0, 0
        for x_real in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
            if args.data_flag == True: # 'mnist' or 'fashion_mnist':
                x_real = x_real[0].to(device) # x_real[1] is flag
            else:
                x_real = x_real.to(device)

            set_seed(seed_flag)
            z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
            seed_flag = seed_flag + 1

#--------training D-----------
            x_fake = G(z) #G(z)[8]
            x_real_d_logit = D(x_real) # D(x_real)[0]
            x_fake_d_logit = D(x_fake.detach())

            x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)

            gp = torch.tensor(0.0)
            #gp = g_penal.gradient_penalty(functools.partial(D), x_real, x_fake.detach(), gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)
            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()

            D_loss_dict={'d_loss': x_real_d_loss.item() + x_fake_d_loss.item(), 'gp': gp.item()}

            it_d += 1
            for k, v in D_loss_dict.items():
                writer.add_scalar('D/%s' % k, v, global_step=it_d+ep*len(data_loader))

#-----------training G-----------
            x_fake_d_logit_2 = D(x_fake)
            G_loss = g_loss_fn(x_fake_d_logit_2) #渐进式loss
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            it_g += 1
            G_loss_dict = {'g_loss': G_loss.item()}
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v, global_step=it_g+ep*len(data_loader))

#-----------training D2E-----------
            # D2E_loss = mse_loss(x_fake,x_true)
            # D_optimizer.zero_grad()
            # D2E_loss.backward()
            # D_optimizer.step()

            # it_g += 1
            # D2E_loss_dict = {'d2e_loss': D2E_loss.item()}
            # for k, v in D2E_loss_dict.items():
            #     writer.add_scalar('G/%s' % k, v, global_step=it_d+ep*len(data_loader))



#--------------save---------------
            if it_g%100==0:
                with torch.no_grad():
                    if args.BN == True:
                        x_sample = sample(z)
                    else:
                        x_sample = x_fake
                    torchvision.utils.save_image(x_fake*0.5+0.5,sample_dir+'/ep%d_it%d.jpg'%(ep,it_g), nrow=int(np.sqrt(args.batch_size)))
                    with open(output_dir+'/loss.txt','a+') as f:
                        print('ep_%d_iter_%d'%(ep,it_g),file=f)
                        print('G_loss:'+str(G_loss.item())+'------'+'D_loss'+str(D_loss.item()),file=f)

        # save checkpoint
        if (ep+1)%10==0:   
            torch.save(G.state_dict(), ckpt_dir+'/Epoch_G_%d.pth' % ep)
            torch.save(D.state_dict(), ckpt_dir+'/Epoch_D_%d.pth' % ep)

            ## visual conv
            # with torch.no_grad():
            #     z = D(x_real)
            #     x = G(z)
            #     x_ = torch.cat((x,x_real))
            #     z_ = D(x)
            #     x__ = G(z_)
            #     x__ = torch.cat((x_,x__))
            #     img_grid = torchvision.utils.make_grid(x_, normalize=True, scale_each=True, nrow=args.batch_size)  # B，C, H, W
            #     writer.add_image('real_img_%d'%(ep), img_grid)

            # #G
            # for name, layer in G.net._modules.items():
            #     z = layer(z)
            #     if isinstance(layer, torch.nn.ConvTranspose2d):
            #         #print(z.shape)
            #         x1 = z.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            #         img_grid = torchvision.utils.make_grid(x1, normalize=True, scale_each=True, nrow=int(np.sqrt(args.batch_size)))  # B，C, H, W
            #         writer.add_image('feature_maps_G_%d_%s'%(ep,name), img_grid)
            #         #torchvision.utils.save_image(x1,'feature_maps%s.png'%name, nrow=100)

            # #D
            # x = z
            # for name, layer in D.net._modules.items():
            #     x = layer(x)
            #     if isinstance(layer, torch.nn.Conv2d):
            #         x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            #         img_grid = torchvision.utils.make_grid(x1, normalize=True, scale_each=True, nrow=int(np.sqrt(args.batch_size)))  # B，C, H, W
            #         writer.add_image('feature_maps_D_%d_%s'%(ep,name), img_grid)
            #         #torchvision.utils.save_image(x1,'./D_feature_maps%s.png'%name, nrow=20)
