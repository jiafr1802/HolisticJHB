# Definition of PoseNet
# author: ynie
# date: March, 2020

import torch
import torch.nn as nn
from models.registers import MODULES
from models.modules import resnet
from models.modules.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from models.jhb.modules.relation_net import RelationNet
from configs.data_config import NYU40CLASSES


@MODULES.register_module
class HJHBNet(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(HJHBNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        # set up neural network blocks
        self.resnet = nn.DataParallel(resnet.resnet34(pretrained=False))

        # set up relational network blocks
        self.relnet = RelationNet()

        # branch to predict the size
        #welcome
        self.fc1 = nn.Linear(4096, 128)  # jfr
        self.fc2 = nn.Linear(128, 3)

        #welcome to explore
        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        # initialize resnet weights
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
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
        print('before resnet: ',x.shape)
        a_features = self.resnet(x)
        print('after resnet: ',a_features.shape)
        '''
        这里需要修改，不然数据维度对应不上～
        经历一个resnet应该OK
        看看dataloader那里有什么经历
        '''
        a_features = a_features.view(a_features.size(0), -1)
        print('after view: ',a_features.shape)
        # branch to predict the size
        cls = self.fc1(a_features)
        cls = self.relu_1(cls)#explore
        cls = self.dropout_1(cls)#explore
        cls = self.fc2(cls)#jfr
        return cls
