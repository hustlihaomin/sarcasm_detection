import numpy as np
import json
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from data.split import split_train_valid_test

origin_path = "/home/data/zhuriyong/Documents/JupyterProjects/SarcasmDetection/resource/dataset/"
feature_file = {
    'GEN': origin_path + 'sarcasm_v2/GEN-sarc-notsarc-feature.csv',
    'HYP': origin_path + 'sarcasm_v2/HYP-sarc-notsarc-feature.csv',
    'RQ': origin_path + 'sarcasm_v2/RQ-sarc-notsarc-feature.csv'
}
feature_path = {
    'GEN': origin_path + 'sarcasm_v2/GEN-sarc-notsarc-feature.npy',
    'HYP': origin_path + 'sarcasm_v2/HYP-sarc-notsarc-feature.npy',
    'RQ': origin_path + 'sarcasm_v2/RQ-sarc-notsarc-feature.npy'
}


class SarcasmDataset(Dataset):
    def __init__(self, dataset, mode='train', need_padding=False):
        self.dataset = dataset
        self.mode = mode
        self.time_length, self.input_dimensions = 0, 0
        if need_padding:
            self._init_sarcasm_corpus_v2()
        else:
            self._load_sarcasm_corpus()

    def _init_sarcasm_corpus_v2(self):
        data = pd.read_csv(feature_file[self.dataset])
        data.loc[data['class'] == 'notsarc', 'class'] = 0
        data.loc[data['class'] == 'sarc', 'class'] = 1
        self.labels, self.indexes = np.array(data['class']), np.array(data['id'])
        self.total_length, self.notsarc_length, self.sarc_length = len(self.labels), len(
            data.loc[data['class'] == 0]), len(
            data.loc[data['class'] == 1])
        self.features = np.array(data['feature'])
        time_lengths = []
        for i in range(len(self.features)):
            self.features[i] = np.array(json.loads(self.features[i]))
            time_lengths.append(self.features[i].shape[0])
        self.time_length = int(np.mean(time_lengths) + 3 * np.std(time_lengths))
        self.input_dimensions = self.features[0].shape[1]
        # 统一时间步长度，不足的补零，遵循 3 sigma原则
        for i in range(len(self.features)):
            if self.features[i].shape[0] < self.time_length:
                paddings = np.zeros((self.time_length - self.features[i].shape[0], 768), )
                self.features[i] = np.concatenate((self.features[i], paddings), axis=0)
            elif self.features[i].shape[0] > self.time_length:
                self.features[i] = self.features[i][0:self.time_length][:]
            else:
                pass
        results = split_train_valid_test(self.features, self.labels)
        np.save(feature_path[self.dataset], results)

    def _load_sarcasm_corpus(self):
        data = np.load(feature_path[self.dataset], allow_pickle=True)
        self.features = np.array(data.item()[self.mode]['features'].tolist())
        self.labels = np.array(data.item()[self.mode]['labels'].tolist())
        self.time_length = self.features[0].shape[0]
        self.input_dimensions = self.features[0].shape[1]

    def get_data(self):
        return self.features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {
            "text": torch.Tensor(self.features[item]),
            "label": torch.Tensor(self.labels[item].reshape(-1)).long()
        }


def sarcasm_dataloader(params):
    datasets = {
        'train': SarcasmDataset(params.dataset, 'train'),
        'valid': SarcasmDataset(params.dataset, 'valid'),
        'test': SarcasmDataset(params.dataset, 'test')
    }
    return {
        "train": DataLoader(
            datasets['train'], batch_size=params.batch_size['train'], num_workers=params.num_workers,
            shuffle=params.shuffle
        ),
        "valid": DataLoader(
            datasets['valid'], batch_size=params.batch_size['valid'], num_workers=params.num_workers,
            shuffle=params.shuffle
        ),
        "test": DataLoader(
            datasets['test'], batch_size=params.batch_size['test'], num_workers=params.num_workers,
            shuffle=params.shuffle
        ),
    }


# print(data)
if __name__ == '__main__':
    model = SarcasmDataset(
        dataset='GEN',
        mode='train',
        need_padding=False
    )
    features = model.get_data()
    print(features.shape)
