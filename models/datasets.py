import os
from torch.utils.data import Dataset
import json

class STREET(Dataset):
    def __init__(self, config, mode):
        '''
        initiate STREET dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '_set_id.json')
        with open(split_file) as file:#打开train.json文件
            self.split = json.load(file)#存储路径吗？是的，从 file_path = self.split[index]file_path = self.split[index]推断
    def __len__(self):
        return len(self.split)