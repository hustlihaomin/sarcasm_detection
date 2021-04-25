import os

import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from operator import methodcaller
from tqdm import tqdm


def metrics_computation(predictions, labels):
    # 指标有P/R/F, 1 denotes 'sarcasm' while 0 denotes 'not sarcasm'
    #             | real = 1 | real = 0
    # predict = 1 | TP       | FP
    # predict = 0 | FN       | TN
    # the formula for accuracy/precision/recall as follows:
    # accuracy = (TP + TN) / (TP + FP + FN + TN)
    # precision:
    #   for sarcasm: precision_sarc= TP / (TP + FP)
    #   for not sarcasm: precision_notsarc = TN / (TN + FN)
    # recall:
    #   for sarcasm: recall_sarc = TP / (TP + FN)
    #   for not sarcasm: recall_notsarc = TN /(FP + TN)
    # F1 score:
    #   for sarcasm: f1_sarc = 2 * precision_sarc * recall_sarc / (precision_sarc + recall_sarc)
    #   for not sarcasm: f1_notsarc = 2 * precision_notsarc * recall_notsarc / (precision_notsarc + recall_notsarc)
    predict_labels, true_labels = np.argmax(predictions, 1),  np.squeeze(labels)
    metric_results = classification_report(true_labels, predict_labels, target_names=[0, 1], output_dict=True)
    # accuracy, weight_accuracy = metric_results['accuracy'], metric_results['weighted avg']
    return metric_results


class Train:
    def __init__(self, input_dimensions, input_length, output_classes, net, log_path, model_path, devices,
                 model_name, dataset, early_stop):
        self.output_classes = output_classes
        self.height, self.width = input_length, input_dimensions
        self.network = net
        self.log_path, self.model_save_path = log_path, model_path
        self.devices = devices

        self.criterion = nn.CrossEntropyLoss()  # input need to be set into (N, C)

        self.writer = self._log_init()

        # 子类需要确定的参数
        self.model_name, self.dataset, self.early_stop = model_name, dataset, early_stop

    def _log_init(self):
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
        os.makedirs(self.log_path)
        return SummaryWriter(self.log_path)

    def train(self, dataloader, parameters, scheduler=None, learning_rate=1e-5, weight_decay=1e-2, patience=10):
        """
        训练过程
        :param dataloader: 数据加载
        :param parameters: 待训练参数
        :param scheduler： 学习率变化，若为None，则不变化，否则输入变化方式
        :param learning_rate: 学习率
        :param weight_decay: 权重衰减
        :param patience: 早停
        :return: 训练结果
        """
        global scheduler_step
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        if scheduler is not None:
            scheduler_step = methodcaller(scheduler)(optimizer, mode='max', factor=0.1, verbose=True, patience=patience)

        best_accuracy, epochs, best_epoch = 0, 0, 0
        train_accuracy, valid_accuracy, train_loss, valid_loss = [], [], [], []
        while True:
            epochs += 1
            self.network.train()
            running_loss, running_accuracy = 0.0, 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    texts, labels = self.data_preparation(batch_data)
                    outputs = self.network(texts)
                    optimizer.zero_grad()
                    loss = self.loss_computation(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # loss输出，每个epoch输出一次
                    running_loss += loss.item()
                    running_accuracy += metrics_computation(
                        outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
                    )['accuracy']

            train_loss.append(running_loss / len(dataloader['train']))
            train_accuracy.append(running_accuracy / len(dataloader['train']))
            print(
                "TRAIN - (model: %s) (difference: %d/epochs: %d)>> loss: %.4f accuracy: %.4f"
                % (self.model_name, epochs - best_epoch, epochs, train_loss[-1], train_accuracy[-1])
            )
            temp_loss, temp_accuracy, _ = self.test(dataloader['valid'])
            valid_loss.append(temp_loss)
            valid_accuracy.append(temp_accuracy)
            print(
                "VALID - (model: %s) (epochs: %d)>> loss: %.4f accuracy: %.4f"
                % (self.model_name, epochs, valid_loss[-1], valid_accuracy[-1])
            )
            if scheduler is not None and patience > 0:
                scheduler_step.step(valid_accuracy[-1])
            if valid_accuracy[-1] > best_accuracy:
                best_accuracy, best_epoch = valid_accuracy[-1], epochs
                model_path = os.path.join(self.model_save_path, f'{self.model_name}-{self.dataset}.pth')
                if os.path.exists(model_path):
                    os.remove(model_path)
                torch.save(self.network.cpu().state_dict(), model_path)
                self.network.to(self.devices)
            # 记录日志
            self.summary_log(train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1], epochs)
            if epochs - best_epoch >= self.early_stop:
                return

    def test(self, dataloader):
        """
        测试过程
        :param dataloader: 预测结果
        :return: 返回测试结果
        """
        self.network.eval()
        test_loss, test_accuracy, results = 0.0, 0.0, None
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    texts, labels = self.data_preparation(batch_data)
                    outputs = self.network(texts)
                    test_loss += self.loss_computation(outputs, labels).item()
                    result_accuracy = metrics_computation(
                        outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
                    )
                    if results is None:
                        results = result_accuracy
                    else:
                        for key, value in result_accuracy.items():
                            if type(value) is dict:
                                for sub_key, sub_value in value.items():
                                    results[key][sub_key] += sub_value
                            else:
                                results[key] += value
            for key, value in results.items():
                if type(value) is dict:
                    for sub_key, sub_value in value.items():
                        results[key][sub_key] /= len(dataloader)
                else:
                    results[key] /= len(dataloader)
            test_loss = test_loss / len(dataloader)
        return test_loss, results['accuracy'], results

    def data_preparation(self, batch_data):
        """
        network输入准备
        :param batch_data:
        :return: (texts, labels)
        """
        return batch_data['text'].to(self.devices), batch_data['label'].to(self.devices)

    def loss_computation(self, outputs, targets):
        """
        计算损失
        :param outputs: network输出结果
        :param targets: 标签
        :return: loss
        """
        return self.criterion(outputs, targets.squeeze())

    def summary_log(self, train_loss, train_accuracy, valid_loss, valid_accuracy, epochs):
        self.writer.add_scalars('loss', {
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, epochs)
        self.writer.add_scalars('accuracy', {
            'train_accuracy': train_accuracy,
            'valid_accuracy': valid_accuracy
        }, epochs)
