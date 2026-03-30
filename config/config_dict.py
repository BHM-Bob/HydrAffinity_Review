'''
self info removed
'''
from mbapy.file import opts_file

class Config(object):
    def __init__(self, config: str, train=True):
        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'
           
        v_num = config.split('_')[0]
        if v_num in ['v1', 'v2', 'v2.1', 'v3', 'v4', 's1', 's2', 't1']:
            root = f'config/{v_num}'
            config_path = f'{root}/{config}.yml'
        else:
            root = 'config'
            config_path = f'{root}/{config}.json'

        if self.mode == 'train':
            self.train_config = opts_file(config_path, way = config_path.split('.')[-1])['train']
        elif self.mode == 'test':
            self.test_config = opts_file(config_path, way = config_path.split('.')[-1])['test']

    def get_mode(self):
        return self.mode

    def get_config(self):
        if self.mode == 'train':
            return self.train_config
        elif self.mode == 'test':
            return self.test_config

    def show_config(self, train=True):
        print('='*50)
        if self.mode == 'train':
            for key, value in self.train_config.items():
                print(f'{key}: {value}')
        elif self.mode == 'test': 
            for key, value in self.test_config.items():
                print(f'{key}: {value}')
        print('='*50)

if __name__ == '__main__':
    # demo
    config = Config('v1_EHIGN_l3h512')
    args = config.get_config()
