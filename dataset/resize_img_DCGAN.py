from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
import torchvision
import skimage

#--------文件夹内的图片转换为tensor:[n,c,h,w]------------------
data_path1 = './CelebA-HQ-img/'
data_path2 = './CelebA-HQ-img-256x256-10_000/'

img_size = 256

for i in range(10_000):
    i_path_1 = data_path1+ str(i)+'.jpg'
    i_path_2 = data_path2+ str(i).rjust(5,'0')+'.jpg'
    image_i = Image.open(i_path_1)
    img = image_i.resize((img_size,img_size))
    img.save(i_path_2)
    print(i_path_1,i_path_2)

# #PIL 2 Tensor
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         #torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])

# images = []
# for idx, image_path in enumerate(imgs_path):
#     img = Image.open(image_path).convert("RGB")
#     img = transform(img)
#     images.append(img)

# imgs_tensor = torch.stack(images, dim=0)
# torchvision.utils.save_image(imgs_tensor,'./imgs_tensor.png',nrow=5)

#-----------------------------------------metric imgs_tensor--------------------------------
# image1 = Image.open('./experiment/with TGAN//ours/2-3.png')
# image1 = image1.resize((64,64))
# array1 = np.array(image1)
# #print(image1.get_shape())

# image2 = Image.open('./experiment/with TGAN/baseline_2.png')
# image2 = image2.resize((64,64))
# array2 = np.array(image2)

# import skimage
# import lpips

# loss_mse = torch.nn.MSELoss()
# loss_lpips = lpips.LPIPS(net='vgg')

# def cosineSimilarty(imgs1_cos,imgs2_cos):
#     values = imgs1_cos.dot(imgs2_cos)/(torch.sqrt(imgs1_cos.dot(imgs1_cos))*torch.sqrt(imgs2_cos.dot(imgs2_cos))) # [0,1]
#     return values


# def metrics(img_tensor1,img_tensor2):

#     psnr = skimage.measure.compare_psnr(img_tensor1.float().numpy(), img_tensor2.float().numpy(), 255)

#     ssim = skimage.measure.compare_ssim(img_tensor1.float().numpy().transpose(1,2,0), img_tensor2.float().numpy().transpose(1,2,0), data_range=255, multichannel=True)#[h,w,c]

#     lpips_value = loss_lpips(img_tensor1.unsqueeze(0),img_tensor2.unsqueeze(0)).mean().detach().numpy()

#     mse_value = loss_mse(img_tensor1,img_tensor2).numpy()/255.0

#     cosine_value = cosineSimilarty(img_tensor1.view(-1),img_tensor2.view(-1)).numpy()

#     print('-------------')
#     print('psnr:',psnr)
#     print('-------------')
#     print('ssim:',ssim)
#     print('-------------')
#     print('lpips:',lpips_value)
#     print('-------------')
#     print('mse:',mse_value)
#     print('-------------')
#     print('cosine:',cosine_value)

#     return psnr, ssim, lpips_value, mse_value, cosine_value

# num = len(imgs_tensor)
# n = 0
# psnr_values = 0
# ssim_values = 0
# lpips_values = 0
# mse_values = 0
# cosine_values = 0

# for i,j in zip(imgs_tensor,imgs_tensor):
#     i = i*255.0
#     j = j*255.0 - 0.01
#     n = n + 1

#     psnr, ssim, lpips_value, mse_value, cosine_value = metrics(i,j)
#     psnr_values +=psnr
#     ssim_values +=ssim
#     lpips_values +=lpips_value
#     mse_values +=mse_value
#     cosine_values +=cosine_value

#     print('img_num:%d--psnr:%f--ssim:%f--lpips_value:%f--mse_value:%f--cosine_value:%f'\
#         (n, psnr_values/n, ssim_values/n, lpips_values/n, mse_values/n, cosine_values/n))

#----------------使用inpaiting 除去图片的右下角的水滴---------
    #-------------cv2实现inpanting---------
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2

# num = 16
# path = './%d.png'%num
# path2 = './%s.png'%str(num).rjust(3,'0') #文件名数字编辑，向左补齐3个0 (1->001,2->002,10->010)


# img = cv2.imread(path) # [h,w,c]
# size = img.shape[0]
# margin = 60

# mask = np.zeros((size,size),np.uint8)  # mask, 必须是np.unit8位
# mask[size-margin:size,size-margin:size]=1

# dst_TELEA = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
# dst_NS = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
# cv2.imwrite(path2,dst_TELEA) #这个效果更好
#cv2.imwrite('t1-2.png',dst_NS)

##-------------change imgesName: 1,2,3-> 00001,00002,00003
# import os
# import cv2

# path = os.listdir('./')
# #print(path)

# for i,j in enumerate(path):
#     if j.endswith('.png'):
#         img = cv2.imread(j)
#         cv2.imwrite('%s.png'%str(i+9).rjust(5,'0'),img

## omit RGB layers from Pytorch pre-trained model
# if args.checkpoint_dir_E != None:
#     E_dict = torch.load(args.checkpoint_dir_E,map_location=torch.device(device))
#     new_state_dict = OrderedDict()
#     for (i1,j1),(i2,j2) in zip (E.state_dict().items(),E_dict.items()):
#             new_state_dict[i1] = j2 
#     E.load_state_dict(new_state_dict)




