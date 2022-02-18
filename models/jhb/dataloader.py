import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from models.datasets import STREET
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import collections
import pickle
default_collate = torch.utils.data.dataloader.default_collate

#专属welcome = 未完待续
HEIGHT_PATCH = 512 # jfr
WIDTH_PATCH = 1024 # jfr
MEAN= [0.61935705, 0.5619458, 0.4853252] # welcome
STD= [0.26332706, 0.24720769, 0.23192036] # welcome
#点解要重新设置图像的高度和宽度？
#点解要归一化 （从函数功能上看，是直接归一为[-1，1]之间吗）
#有好多可以探索的，你就不会感觉无聊啦
'''
这里需要修改！！！！！！
'''
data_transform= transforms.Compose([
    transforms.Resize((HEIGHT_PATCH,WIDTH_PATCH)),
    transforms.ToTensor(),
])

class Beijing_Dataset(STREET):
    def __init__(self,cfg,mode):
        self.cfg=cfg
        super(Beijing_Dataset,self).__init__(cfg.config, mode)
    def __getitem__(self, index):
        file_path=self.split[index]
        #path=os.path.split(file_path)#os.path.split 分开文件路径和文件名 path[0]是文件路径 [1]是文件名
        #file_name=path[1]#文件名
        self.cfg.log_string("="*40)
        self.cfg.log_string("sample file_path: %s " % file_path)  # added by jfr
        with open(file_path, 'rb') as f:#要清楚转换为pkl的好处之后再这样做
            sequence=pickle.load(f)
        #写一版直接加载图像的 ok ! 建议参考计算机视觉作业
        image = Image.fromarray(sequence['rgb_img'])
        image = data_transform(image)#jfr
        label = sequence['label']
        cls_codes = torch.zeros(3)#jfr
        cls_codes[label] = 1#jfr
        #return中包含file_path是为了test的时候有的放矢针对性处理测试结果
        #welcome: 需要转换成one hot的形式，之前用拓扑的形式分析过，现在可以加上空间思维，在其中随便哪一个步骤去做修改都可以
        return {'image': image, 'label': label, 'file_path': file_path}#jfr

def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem

def collate_fn(batch):
    """
    Data collater.
    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:#['image','boxes_batch',...]
        if key == 'boxes_batch':#或许因为涉及到嵌套的问题，我推测是，看一看
            collated_batch[key] = dict()#collated_batch['boxes_batch']={}
            for subkey in batch[0][key]: # subkey is in ['patch', 'g_feature',...]
                if subkey == 'mask':#you can ignore that surely
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    tensor_batch = torch.cat(list_of_tensor) # torch.cat ??? -> explore 推测是从列表（列表元素也是tensor）转换为tensor的形式
                collated_batch[key][subkey] = tensor_batch
        else:#e.g. image 没有嵌套问题，直接安排
            collated_batch[key] = default_collate([elem[key] for elem in batch])

    #interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
    #collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    return collated_batch

def Beijing_dataloader(cfg, mode='train'):#jfr
    config=cfg.config
    dataloader = DataLoader(dataset=Beijing_Dataset(cfg, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn)
    return dataloader

