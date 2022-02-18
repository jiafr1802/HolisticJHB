# Demo script
# author: ynie
# date: April, 2020
from net_utils.utils import load_device, load_model#welcome
from net_utils.utils import CheckpointIO#welcome
from configs.config_utils import mount_external_config#welcome
import numpy as np
import torch
from torchvision import transforms
import os
from time import time
from PIL import Image
import json
import math

from models.jhb.dataloader import collate_fn#welcome

HEIGHT_PATCH = 512#welcome
WIDTH_PATCH = 1024#welcome

data_transforms = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor()
])#welcome

def load_demo_data(demo_path, device):
    img_path = os.path.join(demo_path, 'img.jpg')
    assert os.path.exists(img_path)
    '''preprocess'''
    image = Image.open(img_path).convert('RGB')
    # get object images
    image = data_transforms(image)
    """assemble data"""
    data = collate_fn([{'image':image}])
    image = data['image'].to(device)
    input_data = {'image':image}
    return input_data

def run(cfg):
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)#link

    '''Mount external config data'''
    cfg = mount_external_config(cfg)#link

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)#link

    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)#link
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    checkpoint.parse_checkpoint()
    cfg.log_string('-' * 100)

    '''Load data'''
    cfg.log_string('Loading data.')
    data = load_demo_data(cfg.config['demo_path'], device)

    '''Run demo'''
    net.train(cfg.config['mode'] == 'train')
    with torch.no_grad():
        start = time()
        result = net(data)
        end = time()

    print('Time elapsed: %s.' % (end-start))
    print('result is: ',result)

