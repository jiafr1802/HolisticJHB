# Definition of PoseNet
# author: ynie
# date: March, 2020

import torch
import torch.nn as nn
from models.registers import MODULES
from models.modules import resnet
from models.modules.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from models.jhb.modules.ml_decoder import MLDecoder


@MODULES.register_module
class HJHBNet_ML(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(HJHBNet_ML, self).__init__()
        print('')
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        # set up neural network blocks
        self.resnet = nn.DataParallel(resnet.resnet50(pretrained=False))

        # set up ml-decoder
        self.mldecoder=MLDecoder(num_classes=3)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        # initialize resnet weights
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def forward(self, x):
        '''
        Extract relational features for object bounding box estimation.

        The definition of 'batch' in train.py indicates the number of images we process in a single forward broadcasting.
        In this implementation, we speed-up the efficiency by processing all objects in a batch in parallel.
        :param x: Patch_size x Channel_size x Height x Width
        :return: .the result of the classifier
                Patch_size x 3
        是否下面接的就是loss?
        '''
        # get appearance feature from resnet.
        a_features = self.resnet(x)
        '''
        [bs,2048,4,8]
        '''
        #print('after resnet: ', a_features.shape)

        cls = self.mldecoder(a_features)

        return cls