import os
import lpips
import torch
import numpy as np
import torchvision
from networks.PGGAN import  Encoder , Networks as net
from utils.data_tools import DatasetFromFolder
from torch.autograd import Variable

#----------------Parm-----------------------
using_Dw = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device = 'cuda'
G_Path = './checkpoints/GAN_G.pth' # GAN_GEN_SHADOW_8.pth
D_Path = './checkpoints/GAN_D.pth' # GAN_DIS_8.pth
E_Path = None
batch_size = 4

#----------------path setting---------------
resultPath = "./output/D2E_PGGAN"
if not os.path.exists(resultPath):
	os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
	os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/checkpoints"
if not os.path.exists(resultPath1_2):
	os.mkdir(resultPath1_2)


#----------------test pre-model output-----------
def toggle_grad(model, requires_grad):
	for p in model.parameters():
		p.requires_grad_(requires_grad)

netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512]
netG.load_state_dict(torch.load(G_Path,map_location=device)) #shadow的效果要好一些 

netE = torch.nn.DataParallel(Encoder.encoder(height=9, feature_size=512))
#netE.load_state_dict(torch.load(E_Path,map_location=device))

#--------------Reusing pre-trained D's weight---------
if using_Dw:
	netD = torch.nn.DataParallel(net.Discriminator(height=9, feature_size=512)) # in: [-1,3,1024,1024],out:[]
	netD.load_state_dict(torch.load(D_Path,map_location=device))
	toggle_grad(netD,False)
	toggle_grad(netE,False)
	paraDict = dict(netD.named_parameters()) # pre_model weight dict
	for i,j in netE.named_parameters():
		if i in paraDict.keys():
			w = paraDict[i]
			j.copy_(w)
	toggle_grad(netE,True)
	del netD

#------------------dataSet-----------
# data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
# trans = torchvision.transforms.ToTensor()
# dataSet = DatasetFromFolder(data_path,transform=trans)
# data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=10,shuffle=True,num_workers=4,pin_memory=True)
# image = next(iter(data))
# torchvision.utils.save_image(image, './1.jpg', nrow=1)

# --------------training with generative image------------share weight: good result!------------step2:no share weight:

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
optimizer = torch.optim.Adam(netE.parameters(), lr=0.0015 ,betas=(0.5, 0.99), eps=1e-8)
loss_l2 = torch.nn.MSELoss()
loss_kl = torch.nn.KLDivLoss() #衡量分布
loss_l1 = torch.nn.L1Loss() #稀疏
loss_all=0
for epoch in range(20):
	for i in range(5001):
		z = torch.randn(batch_size, 512).to(device)
		with torch.no_grad():
			x = netG(z,depth=8,alpha=1) # depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
		z_ = netE(x.detach(),depth=8,alpha=1) 
		z_ = z_.squeeze(2).squeeze(2)
		x_ = netG(z_,depth=8,alpha=1)
		optimizer.zero_grad()
		loss_1_1 = loss_fn_vgg(x, x_).mean()
		loss_1_2 = loss_l2(x,x_)
		loss_2 = loss_l2(z,z_)
		loss_i = loss_1_1+loss_1_2+0.01*loss_2
		loss_i.backward()
		optimizer.step()
		loss_all +=loss_i.item()
		print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
		if i % 100 == 0: 
			img = (torch.cat((x[:4],x_[:4]))+1)/2
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=4)
			# with open(resultPath+'/Loss.txt', 'a+') as f:
			# 	print(str(epoch)+'-'+str(i)+'-'+'loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()),file=f)
				#print(str(epoch)+'-'+str(i)+'-'+'loss_1:  '+str(loss_1.item())+'  loss_2:  '+str(loss_2.item())+'  loss_3:  '+str(loss_3.item()),file=f)
				#print(str(epoch)+'-'+str(i)+'-'+'loss_1-1:  '+str(loss_1_1.item())+'  loss_1-2:  '+str(loss_1_2.item())+'  loss_1-3:  '+str(loss_1_3.item()),file=f)
			# with open(resultPath+'/D_z.txt', 'a+') as f:
			# 	print(str(epoch)+'-'+str(i)+'-'+'D_z:  '+str(z_[0,0:30])+'     D_z:    '+str(z_[0,30:60]),file=f)
			# 	print(str(epoch)+'-'+str(i)+'-'+'D_z_mean:  '+str(z_.mean())+'     D_z_std:    '+str(z_.std()),file=f)
	torch.save(netE.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)




















