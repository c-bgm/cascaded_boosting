from VAEBernoulli import VAEBernoulli
import torch
import torch.nn as nn
import torch.nn.functional as F
from VAEBasic import VAEBasic
import numpy as np
from utils import bernoulli_ce, VAEBernoulli_config, binarize
from dataset import load_mnist_2d, QuickDataset, CelebaDataset


class VAEBernoulliLinearLong(VAEBernoulli):
    def __init__(self, config):
        VAEBernoulli.__init__(self, config)
        self.num_linear = self.config['num_linear']
        self.Q_linears_list = []
        self.Q_bns_list = []
        self.P_linears_list = []
        self.P_bns_list = []
        self.X_dim = int(np.prod(config['X_dim']))

        for i in range(self.num_linear):
            if i == 0:
                in_dim = self.X_dim
            else:
                in_dim = config['h_dim']
            self.Q_linears_list.append(nn.Linear(in_dim, config['h_dim']))
            if config['bn']:
                self.Q_bns_list.append(nn.BatchNorm1d(config['h_dim']))
        self.Q_linears = nn.ModuleList(self.Q_linears_list)
        self.Q_bns = nn.ModuleList(self.Q_bns_list)
        self.Qmu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.Qvar_linear = nn.Linear(config['h_dim'], config['Z_dim'])

        for i in range(self.num_linear):
            if i == 0:
                in_dim = config['Z_dim']
            else:
                in_dim = config['h_dim']
            self.P_linears_list.append(nn.Linear(in_dim, config['h_dim']))
            if config['bn']:
                self.P_bns_list.append(nn.BatchNorm1d(config['h_dim']))
        self.P_linears = nn.ModuleList(self.P_linears_list)
        self.P_bns = nn.ModuleList(self.P_bns_list)
        self.P_linear_final = nn.Linear(config['h_dim'], self.X_dim)

        print(self.P_linears)
        print(self.P_bns)
        print(self.Q_linears)
        print(self.Q_bns)

    def Q(self, X):  # q(z|X) = N(z|mu(X), var(X))
        h = X
        for linear in self.Q_linears:
            h = F.relu(linear(h))
        z_mu = self.Qmu_linear(h)
        z_logvar = self.Qvar_linear(h)  # log var
        return z_mu, z_logvar

    def f(self, z):  # p(X|z) = N(X|f(z), sigma^2 I)
        h = z
        for linear in self.P_linears:
            h = F.relu(linear(h))
        X = F.sigmoid(self.P_linear_final(h))
        return X


if __name__ == "__main__":
    def experiment(num_linear, h_dim, phase, binary=True):
        trX, teX, _, _ = load_mnist_2d('../models/data/mnist/raw')
        if binary:
            trX = binarize(trX)
            teX = binarize(teX)
        valX = trX[50000:]
        trX = trX[:50000]
        tr_dataset = QuickDataset(trX)
        val_dataset = QuickDataset(valX)
        te_dataset = QuickDataset(teX)
        dataset = (tr_dataset, val_dataset, te_dataset)

        config = VAEBernoulli_config(X_dim=(28 * 28,), h_dim=h_dim, Z_dim=20, height=28, width=28, channel=1, sigma=0.5,
                                      epoch=50, max_epoch=100,
                                      sample_dir='exp_result/VAEBernoulliLong/%d_layers' % num_linear,
                                      model_dir='models/VAEBernoulliLong/%d_layers' % num_linear,
                                      model_name='VAEBernoulliLong_%dlayers' % num_linear,
                                      retrain=False, phase=phase)
        config['num_linear'] = num_linear
        config['bn'] = True
        model = VAEBernoulliLinearLong(config)
        elbo = model.build(dataset)
        print("[num_linear {}] [phase {}] [elbo {:.4f}]".format(num_linear, phase, elbo))

    for num_linear, h_dim in [(10, 500), (2, 2500)]:
        experiment(num_linear=num_linear, h_dim=h_dim, phase="train")
