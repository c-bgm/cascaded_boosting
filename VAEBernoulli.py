import torch
import torch.nn as nn
import torch.nn.functional as F
from VAEBasic import VAEBasic
import numpy as np
from utils import bernoulli_ce


class VAEBernoulli(VAEBasic):
    def __init__(self, config):
        VAEBasic.__init__(self, config)

    def Q(self, X):  # Q(z|X) = N(z|mu(X), var(X))
        z_mu, z_logvar = -1, -1
        return z_mu, z_logvar

    def f(self, z):  # P(X|z) = Bernoulli(X|f(z))
        X = -1
        return X

    def loss(self, X):
        # Forward
        mu, logvar = self.Q(X)
        eps = torch.randn(len(X), self.config['Z_dim']).cuda()
        z = mu + torch.exp(logvar / 2) * eps
        X_sample = self.f(z)

        # Loss
        recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False) / len(X)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar, 1))
        loss = recon_loss + (self.config['sigma'] ** 2) * kl_loss
        return recon_loss, kl_loss, loss

    def ELBOX__(self, X, tight=False):
        if tight:
            exp_ce_sum = torch.zeros(len(X)).float().cuda()
            for i in range(self.sample_times):
                eps = torch.randn(len(X), self.config['Z_dim']).cuda()
                sample_X = self.f(eps)
                ce = -bernoulli_ce(X, sample_X)
                exp_ce = torch.exp(ce)
                exp_ce_sum += exp_ce

            exp_ce_mean = exp_ce_sum / self.sample_times
            log_exp_ce_mean = torch.log(exp_ce_mean)
            logpX = torch.mean(log_exp_ce_mean)
            return logpX
        else:
            mu, logvar = self.Q(X)
            eps = torch.randn(len(X), self.config['Z_dim']).cuda()
            z = mu + torch.exp(logvar / 2) * eps
            X_sample = self.f(z)

            # ELBO
            ELBO_log = -F.binary_cross_entropy(X_sample, X, size_average=False) / len(X)
            ELBO_kl = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1. - logvar, 1))
            ELBO = ELBO_log - ELBO_kl
            return ELBO


class VAEBernoulliLinear(VAEBernoulli):
    def __init__(self, config):
        VAEBernoulli.__init__(self, config)

        self.bn = False
        if "bn" in config and config['bn'] == True:
            self.bn = True

        self.Q_linear1 = nn.Linear(int(np.prod(config['X_dim'])), config['h_dim'])
        self.Q_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.Qmu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.Qvar_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        if self.bn:
            self.Q_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.Q_bn2 = nn.BatchNorm1d(config['h_dim'])

        self.P_linear1 = nn.Linear(config['Z_dim'], config['h_dim'])
        self.P_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.P_linear3 = nn.Linear(config['h_dim'], int(np.prod(config['X_dim'])))
        if self.bn:
            self.P_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.P_bn2 = nn.BatchNorm1d(config['h_dim'])

    def Q(self, X):  # q(z|X) = N(z|mu(X), var(X))
        if self.bn:
            h = F.relu(self.Q_bn1(self.Q_linear1(X)))
            h = F.relu(self.Q_bn2(self.Q_linear2(h)))
        else:
            h = F.relu(self.Q_linear1(X))
            h = F.relu(self.Q_linear2(h))
        z_mu = self.Qmu_linear(h)
        z_logvar = self.Qvar_linear(h)  # log var
        return z_mu, z_logvar

    def f(self, z):  # p(X|z) = N(X|f(z), sigma^2 I)
        if self.bn:
            h = F.relu(self.P_bn1(self.P_linear1(z)))
            h = F.relu(self.P_bn2(self.P_linear2(h)))
        else:
            h = F.relu(self.P_linear1(z))
            h = F.relu(self.P_linear2(h))
        X = F.sigmoid(self.P_linear3(h))
        return X


class VAEBernoulliConv(VAEBernoulli):
    def __init__(self, config):
        VAEBernoulli.__init__(self, config)
        channel, height, width = config['X_dim']
        self.num_conv = config['num_conv']
        self.Q_convs_list = []
        self.P_deconvs_list = []
        testx = torch.zeros(1, channel, height, width).cuda()
        print(testx.shape)
        for i in range(self.num_conv):
            if i == 0:
                in_channel = channel
            else:
                in_channel = 32
            Q_conv = nn.Conv2d(in_channel, 32, kernel_size=(4, 4), stride=2, padding=1).cuda()
            self.Q_convs_list.append(Q_conv)
            testx = Q_conv(testx)
            print(testx.shape)
        self.Q_convs = nn.ModuleList(self.Q_convs_list)
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
                out_channel = channel
            else:
                out_channel = 32
            P_deconv = nn.ConvTranspose2d(32, out_channel, kernel_size=(4, 4), stride=2, padding=1).cuda()
            testx = P_deconv(testx)
            print(testx.shape)
            self.P_deconvs_list.append(P_deconv)
        self.P_deconvs = nn.ModuleList(self.P_deconvs_list)
        print(self.Q_convs)
        print(self.P_deconvs)

    def Q(self, X):  # q(z|X) = N(z|mu(X), var(X))
        h = X
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
        for idx, deconv in enumerate(self.P_deconvs):
            if idx < self.num_conv - 1:
                h = F.relu(deconv(h))
            else:
                h = F.sigmoid(deconv(h))
        X = h
        return X


if __name__ == '__main__':
    pass
