import torch.utils.data as data
from torch.autograd import Variable
import torch
import numpy as np
from PIL import Image
import os
import os.path
import pdb


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImgListLoader(data.Dataset):

    def __init__(self, root, path_img_list, sep="\t", transform=None, target_transform=None,
                 loader=default_loader):
        self.imgs = []
        this_img_list = open(path_img_list)
        for lines in this_img_list:
            self.imgs.append(lines)
        if len(self.imgs) == 0:
            raise(RuntimeError("no image loaded "+ "\n"))
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.sep = sep
        
    # add your own func here
#    def __getitem__(self, index):
#        this_info = self.imgs[index]
#        this_info = this_info.split(" ")
#        path = this_info[0]
#        path = os.path.join(self.root, path)
#        targets = []
#        img = self.loader(path)
#
#        if self.transform is not None:
#            img = self.transform(img)
#        if self.target_transform is not None:
#            targets = self.target_transform(targets)
#        
#            targets = np.array([int(ii) for ii in tt])
#        pdb.set_trace()
#        return img, targets

    def __getitem__(self, index):
        this_info = self.imgs[index]
        this_info = this_info.split(self.sep)
        path = this_info[0]
        path = os.path.join(self.root, path)
        targets = []
        for this_label in range(1,len(this_info)):
            targets.append(int(this_info[this_label]))
            
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        
        targets = np.array(targets) 
        return img, torch.LongTensor(targets)
        #return img, torch.Tensor(targets)
        #return img, targets[0]
    def __len__(self):
        return len(self.imgs)
