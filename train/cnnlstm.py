from train.base import Train


class CNNLSTMTrain(Train):
    def __init__(self, input_dimensions, input_length, output_classes, net, log_path, model_path, devices,
                 dataset, early_stop):
        super().__init__(input_dimensions, input_length, output_classes, net, log_path, model_path, devices, 'CNNLSTM',
                         dataset, early_stop)

    def do_train(self, dataloader, parameters, scheduler=None, learning_rate=1e-5, weight_decay=1e-2, patience=10):
        self.train(dataloader, parameters, scheduler, learning_rate, weight_decay, patience)

    def do_test(self, dataloader, network):
        return self.test(dataloader, network)