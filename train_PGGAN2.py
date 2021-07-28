import os
import lpips
import torch
import numpy as np
import torchvision
import networks.PGGAN_V2.pggan_encoder as E
import networks.PGGAN_V2.pggan_generator as G
import networks.PGGAN_V2.pggan_discriminator as D

#----------------Parm-----------------------
using_Dw = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device = 'cuda'
batch_size = 12
G_Path = './checkpoint/pggan_horse256.pth'

#----------------path setting---------------
resultPath = "./output/RC_Training_D_percp_kl_mse"
if not os.path.exists(resultPath):
	os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
	os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath1_2):
	os.mkdir(resultPath1_2)


#----------------test pre-model output-----------
netG = G.PGGANGenerator(resolution=256).to(device)
checkpoint = torch.load(G_Path) #map_location='cpu'
if 'generator_smooth' in checkpoint: #默认是这个
    generator.load_state_dict(checkpoint['generator_smooth']) #default
else:
    generator.load_state_dict(checkpoint['generator'])

netE = E.PGGAN_Encoder(256, minibatch_std_group_size = batch_size) # out: [n,512]

# --------------training with generative image------------share weight: good result!------------step2:no share weight:

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
optimizer = torch.optim.Adam(netD2.parameters(), lr=0.002 ,betas=(0, 0.99), eps=1e-8)
loss_l2 = torch.nn.MSELoss()
loss_kl = torch.nn.KLDivLoss() #衡量分布
loss_l1 = torch.nn.L1Loss() #稀疏
loss_all=0
for epoch in range(20):
	for i in range(5001):
		z = torch.randn(batch_size, 512).to(device)
		with torch.no_grad():
			x = netG(z,depth=8,alpha=1)
		z_ = netE(x.detach(),height=8,alpha=1)
		z_ = z_.squeeze(2).squeeze(2)
		x_ = netG(z_,depth=8,alpha=1)
		optimizer.zero_grad()
		loss_1_1 = loss_fn_vgg(x, x_).mean()
		loss_1_2 = loss_l2(x,x_)
		y1,y2 = torch.nn.functional.softmax(x_),torch.nn.functional.softmax(x)
		loss_1_3 = loss_kl(torch.log(y1),y2)
		loss_1_3 = torch.where(torch.isnan(loss_1_3), torch.full_like(loss_1_3, 0), loss_1_3)
		loss_1_3 = torch.where(torch.isinf(loss_1_3), torch.full_like(loss_1_3, 1), loss_1_3)
		loss_2 = loss_l2(z.mean(),z_.mean())
		loss_3 = loss_l2(z.std(),z_.std()) 
		loss_1 = loss_1_1+loss_1_2+loss_1_3
		loss_i = loss_1+0.01*loss_2+0.01*loss_3
		loss_i.backward()
		optimizer.step()
		loss_all +=loss_i.item()
		print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
		if i % 100 == 0: 
			img = (torch.cat((x[:4],x_[:4]))+1)/2
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=4)
			with open(resultPath+'/Loss.txt', 'a+') as f:
				print(str(epoch)+'-'+str(i)+'-'+'loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()),file=f)
				print(str(epoch)+'-'+str(i)+'-'+'loss_1:  '+str(loss_1.item())+'  loss_2:  '+str(loss_2.item())+'  loss_3:  '+str(loss_3.item()),file=f)
				print(str(epoch)+'-'+str(i)+'-'+'loss_1-1:  '+str(loss_1_1.item())+'  loss_1-2:  '+str(loss_1_2.item())+'  loss_1-3:  '+str(loss_1_3.item()),file=f)
			with open(resultPath+'/D_z.txt', 'a+') as f:
				print(str(epoch)+'-'+str(i)+'-'+'D_z:  '+str(z_[0,0:30])+'     D_z:    '+str(z_[0,30:60]),file=f)
				print(str(epoch)+'-'+str(i)+'-'+'D_z_mean:  '+str(z_.mean())+'     D_z_std:    '+str(z_.std()),file=f)
	torch.save(netE.state_dict(), resultPath1_2+'/D_model_ep%d.pth'%epoch)




















