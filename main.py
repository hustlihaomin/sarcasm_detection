# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os

import numpy
import random
import torch

from config.CNN_config import ConfigCNN
from config.DNN_config import ConfigDNN
from config.LSTM_config import ConfigLSTM
from data.load_data import sarcasm_dataloader
from models import CNNModule
from models.DNN import DNNModule
from models.textCNN import TextCNNModule
from models.LSTM import LSTMModule
from models.CNN_LSTM import CNNLSTMModule
from models.LSTM_Attention import LSTMAttentionModule
from train.cnn import CNNTrain
from train.dnn import DNNTrain
from train.textcnn import TextCNNTrain
from train.lstm import LSTMTrain
from train.cnnlstm import CNNLSTMTrain
from train.lstmattention import LSTMAttentionTrain

def setup_seed():
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='twitter')
    parser.add_argument('--model_name', type=str, default='CNN')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='./model_path/')
    parser.add_argument('--log_path', type=str, default='./log_path/')
    parser.add_argument(
        '--data_path', type=str, default='/home/zhuriyong/Documents/JupyterProjects/SarcasmDetection/resource/dataset/'
    )
    parser.add_argument('--gpu_ids', type=list, default=[1])
    return parser.parse_args()


def init_model(params):
    using_cuda = len(params.gpu_ids) > 0 and torch.cuda.is_available()
    print("Use %d GPUs!" % len(params.gpu_ids))
    params.devices = torch.device('cuda:%d' % params.gpu_ids[0] if using_cuda else 'cpu')
    dataloader = sarcasm_dataloader(params)
    network = LSTMAttentionModule(
        input_dimensions=params.input_dimensions,
        input_length=params.input_length,
        output_classes=params.output_classes,
    )
    network.to(device=params.devices)

    if using_cuda and len(params.gpu_ids) > 1:
        network = torch.nn.DataParallel(
            network, device_ids=params.gpu_ids, output_device=params.gpu_ids[0]
        )

    train = LSTMAttentionTrain(
        input_dimensions=params.input_dimensions,
        input_length=params.input_length,
        output_classes=params.output_classes,
        net=network,
        log_path=params.log_path, model_path=params.model_path, devices=params.devices, dataset=params.dataset,
        early_stop=params.early_stop
    )

    if params.is_train:
        train.do_train(
            dataloader, network.parameters(), params.scheduler,#params.scheduler
            params.learning_rate, params.weight_decay, params.patience
        )
    pretrained_path = os.path.join(params.model_path, f'{params.model_name}-{params.dataset}.pth')
    print(pretrained_path)
    assert os.path.exists(pretrained_path)
    network.load_state_dict(torch.load(pretrained_path))
    network.to(device=params.devices)
    _, _, results = train.do_test(dataloader['test'], network)
    print(results)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    setup_seed()
    config = ConfigLSTM(parse_args())
    init_model(config.get_config())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
