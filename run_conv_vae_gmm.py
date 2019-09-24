from boosting import *
from VAEBernoulli import VAEBernoulli
from dataset import load_mnist_4d, QuickDataset
import torch.nn as nn
import sys
from MixGauss import MixGauss

lr = 1e-3
batch_size = 256


class VAEBernoulliConvMnist(VAEBernoulli):
    def __init__(self, config):
        VAEBernoulli.__init__(self, config)
        channel, height, width = config['X_dim']

        self.num_conv = config['num_conv']
        self.bn = False
        if "bn" in config and config['bn'] == True:
            self.bn = True
        self.Q_convs_list = []
        self.P_deconvs_list = []
        if self.bn:
            self.Q_bns_list = []
            self.P_bns_list = []
        testx = torch.zeros(1, channel, height, width).cuda()
        print(testx.shape)
        channels = [channel, 32, 64, 128, 256, 256, 256]
        for i in range(self.num_conv):
            if i == 0:
                padding = 3
            else:
                padding = 1
            in_channel = channels[i]
            out_channel = channels[i+1]
            Q_conv = nn.Conv2d(in_channel, out_channel, kernel_size=(4, 4), stride=2, padding=padding).cuda()
            Q_bn = nn.BatchNorm2d(out_channel)
            self.Q_convs_list.append(Q_conv)
            self.Q_bns_list.append(Q_bn)
            testx = Q_conv(testx)
            print(testx.shape)
        self.Q_convs = nn.ModuleList(self.Q_convs_list)
        self.Q_bns = nn.ModuleList(self.Q_bns_list)
        self.middle_shape = testx.shape[1:]
        testx = torch.flatten(testx, 1, -1)
        print(testx.shape)
        self.Q_linear = nn.Linear(testx.shape[1], config['h_dim']).cuda()
        self.Qmu_linear = nn.Linear(config['h_dim'], config['Z_dim']).cuda()
        self.Qvar_linear = nn.Linear(config['h_dim'], config['Z_dim']).cuda()

        self.P_linear1 = nn.Linear(config['Z_dim'], config['h_dim']).cuda()
        self.P_linear2 = nn.Linear(config['h_dim'], testx.shape[1]).cuda()
        testx = testx.view(1, *self.middle_shape)
        print(testx.shape)
        for i in range(self.num_conv):
            if i == self.num_conv - 1:
                padding = 3
            else:
                padding = 1
            in_channel = channels[self.num_conv - i]
            out_channel = channels[self.num_conv - 1 - i]
            P_deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4, 4), stride=2, padding=padding).cuda()
            P_bn = nn.BatchNorm2d(out_channel)
            testx = P_deconv(testx)
            print(testx.shape)
            self.P_deconvs_list.append(P_deconv)
            self.P_bns_list.append(P_bn)
        self.P_deconvs = nn.ModuleList(self.P_deconvs_list)
        self.P_bns = nn.ModuleList(self.P_bns_list)
        print(self.Q_convs)
        print(self.Q_bns)
        print(self.P_deconvs)
        print(self.P_bns)
        sys.stdout.flush()

    def Q(self, X):  # q(z|X) = N(z|mu(X), var(X))
        h = X
        if self.bn:
            for conv, bn in zip(self.Q_convs, self.Q_bns):
                h = F.relu(bn(conv(h)))
        else:
            for conv in self.Q_convs:
                h = F.relu(conv(h))
        h = torch.flatten(h, 1, -1)
        h = F.relu(self.Q_linear(h))
        z_mu = self.Qmu_linear(h)
        z_logvar = self.Qvar_linear(h)  # log var
        return z_mu, z_logvar

    def f(self, z):  # p(X|z) = N(X|f(z), sigma^2 I)
        h = F.relu(self.P_linear1(z))
        h = F.relu(self.P_linear2(h))
        h = h.view(len(h), *self.middle_shape)
        if self.bn:
            for idx, (deconv, bn) in enumerate(zip(self.P_deconvs, self.P_bns)):
                if idx < self.num_conv - 1:
                    h = F.relu(bn(deconv(h)))
                else:
                    h = F.sigmoid(bn(deconv(h)))
        else:
            for idx, deconv in enumerate(self.P_deconvs):
                if idx < self.num_conv - 1:
                    h = F.relu(deconv(h))
                else:
                    h = F.sigmoid(deconv(h))
        X = h
        return X


def experiment(id, phase, binary=True):
    sample_path = 'exp_result/conv_vaes/exp%d' % id
    model_path = 'models/conv_vaes/exp%d' % id

    trX, teX, _, _ = load_mnist_4d('../models/data/mnist/raw')
    if binary:
        trX = binarize(trX)
        teX = binarize(teX)
    valX = trX[50000:]
    trX = trX[:50000]
    tr_dataset = QuickDataset(trX)
    val_dataset = QuickDataset(valX)
    te_dataset = QuickDataset(teX)

    if phase == "train":
        te_dataset = tr_dataset

    microConfigs = []
    microModules = []

    config1 = {'X_dim': (1, 28, 28), 'h_dim': 500, 'Z_dim': 50,
               'sigma': 0.6,
               'num_conv': 4,
               'height': 28, 'width': 28, 'channel': 1,
               'lr': lr, 'bn': True,
               'epoch': 200, 'max_epoch': 200,
               'batch_size': batch_size,
               'sample_dir': os.path.join(sample_path, 'VAEBernoulliConvMnist1'),
               'model_dir': os.path.join(model_path, 'VAEBernoulliConvMnist1'),
               'model_name': 'VAEBernoulliConv1',
               'retrain': False,
               'phase': phase}
    microConfigs.append(config1)
    microModules.append(VAEBernoulliConvMnist)

    config2 = {'X_dim': (50,), 'K': 10,
               'batch_size': batch_size,
               'sample_dir': os.path.join(sample_path, 'MixGauss2'),
               'model_dir': os.path.join(model_path, 'MixGauss2'),
               'model_name': 'MixGauss2',
               'retrain': False,
               'phase': phase}
    microConfigs.append(config2)
    microModules.append(MixGauss)

    config = {'microModules': microModules,
              'microConfigs': microConfigs,
              'dataset': (tr_dataset, val_dataset, te_dataset),
              'sample_dir': os.path.join(sample_path, 'boosting'),
              'phase': phase}

    model = MacroModule(config)
    elbos, elbo_thresholds = model.build()
    display_elbo(elbos, elbo_thresholds)


if __name__ == "__main__":
    experiment(id=0, phase="train")
    experiment(id=0, phase="test")
