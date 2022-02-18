import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'#jfr
import argparse
from configs.config_utils import CONFIG
import train, test, demo

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Total 3D Understanding.')#welcome
    parser.add_argument('config', type=str, default='configs/JHB.yaml',
                        help='configure file for training or testing.')#welcome
    parser.add_argument('--mode', type=str, default='train', help='train, test, or demo')#jfr
    parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')#welcome
    #需要自己去设计和规定一下demo输入的文件的格式、形式
    return parser

#print('lalala')#jfr
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2'#jfr
    parser = parse_args()
    cfg = CONFIG(parser)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)

    '''Run'''
    if cfg.config['mode'] == 'train':
        try:
            train.run(cfg)
        except KeyboardInterrupt:
            pass
        except:
            raise
        #cfg.update_config(mode='test', resume=True, weight=os.path.join(cfg.save_path, 'model_best.pth'))
    if cfg.config['mode'] == 'test':
        test.run(cfg)
    if cfg.config['mode'] == 'demo':
        demo.run(cfg)
