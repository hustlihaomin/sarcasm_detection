import os
import random

from config.storge import Storage


class ConfigCNN:
    def __init__(self, args):
        # 设置数据所在目录
        self.data_path = args.data_path
        # 设置其它参数
        try:
            self.global_params = vars(args)
        except TypeError:
            self.global_params = args

    def __dataset_common_params(self):
        return {
            'gen': {
                'data_path': os.path.join(self.data_path, 'sarcasm_v2/GEN-sarc-notsarc-feature.npy'),
                'input_length': 181,    # TODO need to modify
                'input_dimensions': 768
            },
            'hyp': {
                'data_path': os.path.join(self.data_path, 'sarcasm_v2/HYP-sarc-notsarc-feature.npy'),
                'input_length': 216,  # TODO need to modify
                'input_dimensions': 768
            },
            'rq': {
                'data_path': os.path.join(self.data_path, 'sarcasm_v2/RQ-sarc-notsarc-feature.npy'),
                'input_length': 229,  # TODO need to modify
                'input_dimensions': 768
            },
        }

    @staticmethod
    def __cnn_config_params():
        return {
            'common_params': {
                'batch_size': {
                    'train': 24,
                    'valid': 24,
                    'test': 24
                },
                'num_workers': 8,
                'shuffle': True,
                'output_classes': 2
            },
            # dataset
            'dataset_params': {
                'gen': {
                    'learning_rate': random.choice([5e-3, 1e-3, 1e-4]),
                    'scheduler': 'ReduceLROnPlateau',
                    'weight_decay': 1e-2,
                    'patience': 10,
                    'early_stop': 20,
                },
                'hyp': {
                    'learning_rate': random.choice([5e-3, 1e-3, 1e-4]),
                    'scheduler': 'ReduceLROnPlateau',
                    'weight_decay': 1e-2,
                    'patience': 10,
                    'early_stop': 20,
                },
                'rq': {
                    'learning_rate': random.choice([5e-3, 1e-3, 1e-4]),
                    'scheduler': 'ReduceLROnPlateau',
                    'weight_decay': 1e-2,
                    'patience': 10,
                    'early_stop': 20,
                },
            },
        }

    def get_config(self):
        # normalize
        dataset_name = str.lower(self.global_params['dataset'])
        # integrate all parameters
        return Storage(dict(
            self.global_params,
            **self.__cnn_config_params()['dataset_params'][dataset_name],
            **self.__cnn_config_params()['common_params'],
            **self.__dataset_common_params()[dataset_name]
        ))
