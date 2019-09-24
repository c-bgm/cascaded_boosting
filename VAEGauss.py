from VAEBasic import VAEBasic
import torch
from utils import debug
import abc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAEGauss(VAEBasic):
    def __init__(self, config):
        VAEBasic.__init__(self, config)
        self.sigma = config['sigma']

    @abc.abstractmethod
    def Q(self, X):  # q(z|X) = N(z|mu(X), var(X))
        z_mu, z_logvar = -1, -1
        return z_mu, z_logvar

    @abc.abstractmethod
    def f(self, z):  # p(X|z) = N(X|f(z), sigma^2 I)
        X = -1
        return X

    def loss(self, X):
        mu, logvar = self.Q(X)
        eps = torch.randn(len(X), self.config['Z_dim']).cuda()
        z = mu + torch.exp(logvar / 2) * eps
        X_sample = self.f(z)

        # Loss
        recon_loss = torch.sum((X_sample - X) ** 2) / len(X)
        kl_loss = torch.mean(torch.sum(torch.exp(logvar) + mu ** 2 - logvar, 1))
        loss = recon_loss + (self.sigma ** 2) * kl_loss
        return recon_loss, kl_loss, loss

    def ELBOX__(self, X, tight=False):
        if tight:
            N = np.prod(list(X.shape)[1:])
            exp_diff_sum = torch.zeros(len(X)).float().cuda()
            for i in range(self.sample_times):
                eps = torch.randn(len(X), self.config['Z_dim']).cuda()
                sample_X = self.f(eps)
                diff_square = (X - sample_X) ** 2
                diff = torch.sum(diff_square, list(range(1, len(diff_square.shape))))
                exp_diff = torch.exp(-diff / (2 * (self.sigma ** 2)))
                exp_diff_sum += exp_diff
                exp_diff_sum = exp_diff_sum.detach()
            exp_diff_mean = exp_diff_sum / self.sample_times
            log_exp_diff_mean = torch.log(exp_diff_mean)
            log_exp_diff_mean = torch.mean(log_exp_diff_mean)
            logpX = log_exp_diff_mean - N * np.log(self.sigma) - 0.5 * N * np.log(2 * np.pi)
            return logpX
        else:
            mu, logvar = self.Q(X)
            eps = torch.randn(len(X), self.config['Z_dim']).cuda()
            z = mu + torch.exp(logvar / 2) * eps
            X_sample = self.f(z)

            debug(X[:10])
            debug(X_sample[:10])

            # ELBO
            N = np.prod(list(X.shape)[1:])
            ELBO_log = - (torch.sum((X_sample - X) ** 2) / (2 * self.sigma**2)) / len(X)
            ELBO_log -= 0.5 * N * np.log(2*np.pi)
            ELBO_log -= N * np.log(self.sigma)
            ELBO_kl = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1. - logvar, 1))
            ELBO = ELBO_log - ELBO_kl
            return ELBO


class VAEGaussLinear(VAEGauss):
    def __init__(self, config):
        VAEGauss.__init__(self, config)
        self.bn = False
        if "bn" in config.keys() and config['bn'] == True:
            self.bn = True
        self.Q_linear1 = nn.Linear(int(np.prod(config['X_dim'])), config['h_dim'])
        self.Q_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        if self.bn:
            self.Q_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.Q_bn2 = nn.BatchNorm1d(config['h_dim'])
        self.Qmu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.Qvar_linear = nn.Linear(config['h_dim'], config['Z_dim'])

        self.P_linear1 = nn.Linear(config['Z_dim'], config['h_dim'])
        self.P_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        if self.bn:
            self.P_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.P_bn2 = nn.BatchNorm1d(config['h_dim'])
        self.P_linear3 = nn.Linear(config['h_dim'], int(np.prod(config['X_dim'])))

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
        X = self.P_linear3(h)
        return X


if __name__ == '__main__':
    pass
