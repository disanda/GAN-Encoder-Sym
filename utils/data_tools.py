# coding=utf-8
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class DatasetFromFolder(Dataset):
    def __init__(self,path='',transform=None, channels = 3):
        super().__init__()
        self.channels = channels
        self.path = path
        self.transform = transform
        self.image_filenames = [x for x in os.listdir(self.path) if x.endswith('jpg') or x.endswith('png')] # x.startswith() 
        #imgs_path = os.listdir(path)
        #self.image_filenames = list(filter(lambda x:x.endswith('jpg') or x.endswith('png') ,imgs_path))
    def __getitem__(self, index):
        if self.channels == 1:
            a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('L') # 'L'是灰度图, 'RGB'彩色
        elif self.channels ==3:
            a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('RGB')
        else:
            print('error')
        #a = a.resize((self.size, self.size), Image.BICUBIC)
        #a = transforms.ToTensor()(a)
        if self.transform:
            a = self.transform(a)
        return a
    def __len__(self):
        return len(self.image_filenames)


def make_dataset(dataset_name, batch_size,img_size,img_path=None,drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    transform_RGB = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])
    transform_L = transforms.Compose([
            transforms.Resize(size=(img_size, img_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(), # [0,255] -> [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5]), # [0,1] -> [1,1] (x*mean+std)
            #transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)) # x -> (x,x,x)
            #transfroms.Gradscale(),
        ])
    if dataset_name == 'mnist' or dataset_name=='fashion_mnist':
        if dataset_name == 'mnist':
            dataset = datasets.MNIST(img_path, transform=transform_L, download=False)
        else:
            dataset = datasets.FashionMNIST(img_path, transform=transform_L, download=False)
        img_shape = [img_size, img_size, 1]
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(img_path, transform=transform_RGB, download=True)
        img_shape = [img_size, img_size, 3]
    elif dataset_name == 'STL10':
        dataset = datasets.STL10(root = img_path,transform=transform_RGB, split='unlabeled') #split: 'train', 'test', 'unlabeled'
    elif dataset_name == '3dface':
        dataset = DatasetFromFolder(path=img_path,transform=transform_RGB)
    elif dataset_name == 'celeba':
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]# [image,height,width]
        dataset = DatasetFromFolder(path=img_path,transform=transform_RGB)
    elif dataset_name == 'Celeba_HQ':
        dataset = DatasetFromFolder(path=img_path,transform=transform_RGB,channels=3)
    else:
        raise NotImplementedError
    img_shape = (img_size, img_size, 3)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)
    return data_loader, img_shape

# ==============================================================================
# =                                   debug                                    =
# ==============================================================================

# pose = DatasetFromFolder('/Users/apple/Desktop/AI_code/dataSet/pose_set_1000')

# train_loader = DataLoader(
#      dataset=pose,
#      batch_size=25,#一个batch25张图片,epoch=allData_size/batch_size
#      shuffle=False,
#      #num_workers=0,若是win需要这一行
#      pin_memory=True,#用Nvidia GPU时生效
#      drop_last=True
#  )

# import tqdm
# for x_real in tqdm.tqdm(train_loader, desc='Inner Epoch Loop'):
#     print(type(x_real))

# for i, x in enumerate(train_loader):
#      print(i)
#      print(x.shape)#[n,c,w,h]
#      torchvision.utils.save_image(x, './pose-img/%d.jpg'%(i), nrow=5)#这个保存是三通道的,需要改成1通道
