from boosting import *
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import QuickDataset
from utils import *
import numpy as np
import abc
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import sys


class VAEBasic(MicroModule, nn.Module):
    def __init__(self, config):
        MicroModule.__init__(self, config)
        nn.Module.__init__(self)
        self.sample_times = 4000
        if 'sample_times' in config.keys():
            self.sample_times = config['sample_times']

    @abc.abstractmethod
    def Q(self, X):  # q(z|X) = N(z|mu(X), var(X))
        z_mu, z_logvar = -1, -1
        return z_mu, z_logvar

    @abc.abstractmethod
    def f(self, z):  # E[p(X|z)]
        X = -1
        return X

    @abc.abstractmethod
    def loss(self, X):  #, beta=1.):  # beta is the warm-up coefficie
        recon_loss, kl_loss, loss = -1
        return recon_loss, kl_loss, loss

    @abc.abstractmethod
    # if tight == True, use sample methods to approximate the exact log p(x) (slow)
    # if tight == False, calculate the variational lower bound (Quick)
    def ELBOX__(self, X, tight=False): # ELBO of log p(x)
        return -1

    def ELBOZ__(self, Z): # ELBO of log p(z)
        ELBO = -0.5 * torch.sum(Z**2) / len(Z) - 0.5 * self.config['Z_dim'] * np.log(2*np.pi)
        return ELBO

    def ELBOX(self, X, tight=False):
        if isinstance(X, np.ndarray):
            dataset = QuickDataset(X)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        elif isinstance(X, Dataset):
            dataloader = DataLoader(X, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        else:
            print("[ELBOX] [X type % s]" % type(X))
        ELBOs = []
        for idx, batch in enumerate(dataloader):
            ELBO = self.ELBOX__(batch.cuda(), tight=tight).cpu().detach().numpy()
            ELBOs.append(ELBO)
        return np.mean(ELBOs)

    def ELBOZ(self, Z):
        if isinstance(Z, np.ndarray):
            dataset = QuickDataset(Z)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        elif isinstance(Z, Dataset):
            dataloader = DataLoader(Z, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        else:
            raise Exception("[ELBO] [Z type % s]" % type(Z))
        ELBOs = []
        for batch in dataloader:
            ELBO = self.ELBOZ__(batch.cuda()).cpu().detach().numpy()
            ELBOs.append(ELBO)
        return np.mean(ELBOs)

    def sampleX(self, Z=None):
        if Z is None:
            Z = torch.randn(self.config['batch_size'], self.config['Z_dim']).cuda()
        X = self.f(Z)
        return X

    def sampleZ(self, X=None):
        if X is None:
            eps = torch.randn(self.config['batch_size'], self.config['Z_dim'])
            return eps.detach().numpy()
        elif isinstance(X, np.ndarray):
            dataset = QuickDataset(X)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        elif isinstance(X, Dataset):
            dataloader = DataLoader(X, batch_size=self.config['batch_size'], shuffle=False)
        else:
            raise Exception("[sampleZ] [X type % s]" % (type(X)))
        Zs = []
        for batch in dataloader:
            eps = torch.randn(len(batch), self.config['Z_dim']).cuda()
            mu, log_var = self.Q(batch.cuda())
            Z = mu + torch.exp(log_var / 2) * eps
            Zs.append(Z.cpu().detach().numpy())
        Zs = np.concatenate(Zs)
        assert len(Zs) == len(X)
        return Zs

    def save_samples(self, dataset, ep):
        if not os.path.exists(self.config['sample_dir']):
            os.makedirs(self.config['sample_dir'])
        height, width, channel = self.config['height'], self.config['width'], self.config['channel']

        origin = []
        idxes = list(range(len(dataset)))
        random.shuffle(idxes)
        for i in idxes[:100]:
            origin.append(dataset[i])
        origin = np.array(origin)
        K = 10
        if len(origin) < 100:
            K = int(len(origin)**0.5)
        samples = arr2img(np.clip(origin, 0., 1.), height, width, channel)
        savefig(samples, K, os.path.join(self.config['sample_dir'], 'origin_{}.png'.format(str(ep).zfill(3))))

        X = torch.from_numpy(origin).float().cuda()
        mu, logvar = self.Q(X)
        eps = torch.randn(len(samples), self.config['Z_dim']).cuda()
        z = mu + torch.exp(logvar / 2) * eps

        # reconstruct
        samples = self.f(z).cpu().data.numpy()
        samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
        savefig(samples, K, os.path.join(self.config['sample_dir'], 'reconstruct_{}.png'.format(str(ep).zfill(3))))

        # random sample
        z = torch.randn(100, self.config['Z_dim']).cuda()
        samples = self.f(z).cpu().data.numpy()
        samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
        savefig(samples, 10, os.path.join(self.config['sample_dir'], '{}.png'.format(str(ep).zfill(3))))

    def save_checkpoint(self, ep, elbo, model, optim, path):
        ckpt = {
            'ep': ep,
            'elbo': elbo,
            'model': model,
            'optim': optim,
            'finish': False
        }
        torch.save(ckpt, path)

    def update_checkpoint_ep(self, ep, path):
        ckpt = torch.load(path)
        ckpt['ep'] = ep
        torch.save(ckpt, path)

    def update_checkpoint_finish(self, flag, path):
        ckpt = torch.load(path)
        ckpt['finish'] = flag
        torch.save(ckpt, path)

    def build(self, dataset, elbo_threshold=-10.0e10, save_fig=True):  # train model or load model parameters
        self.cuda()
        model_path = os.path.join(self.config['model_dir'], '%s.pth' % self.config['model_name'])
        tr_dataset, val_dataset, te_dataset = dataset
        tr_dataloader = DataLoader(tr_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)

        ep = 0
        epoch = self.config['epoch']
        solver = optim.Adam(params=self.parameters(), lr=self.config['lr'])
        max_elbo = -10.0e10
        finish = False
        if not self.config['retrain'] and os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.load_state_dict(ckpt['model'])
            solver.load_state_dict(ckpt['optim'])
            ep = ckpt['ep']
            max_elbo = ckpt['elbo']
            finish = ckpt['finish']

        if self.config['phase'] != "train":
            elbo = self.ELBOX(te_dataset).item()
            return elbo

        while not finish and ep < self.config['max_epoch'] and not (ep >= epoch and max_elbo > elbo_threshold):
            for idx, batch in enumerate(tr_dataloader):
                X = batch.cuda()
                recon_loss, kl_loss, loss = self.loss(X)
                loss.backward()
                solver.step()
                solver.zero_grad()
                if 'disp_interval' in self.config and idx % self.config['disp_interval'] == 0:
                    print("[{}] [epoch {}/{}] [iter {}/{}] [recon_loss {}] [kl_loss {}] [loss {}]"
                          .format(self.config['model_name'], ep, epoch,
                                  idx, len(tr_dataloader), recon_loss, kl_loss, loss))
                    sys.stdout.flush()
            if save_fig:
                self.save_samples(te_dataset, ep)

            ep += 1
            # save model
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            elbo = self.ELBOX(val_dataset).item()
            elbo_te = self.ELBOX(te_dataset).item()
            print("[{}] [epoch {}/{}] [elbo_te: {:.2f}] [elbo_val {:.2f}] [max_elbo {:.2f}] [elbo_threshold {:.2f}]"
                  .format(self.config['model_name'], ep, epoch, elbo_te, elbo, max_elbo, elbo_threshold))
            sys.stdout.flush()
            if elbo > max_elbo:
                max_elbo = elbo
                self.save_checkpoint(ep, elbo, self.state_dict(), solver.state_dict(), model_path)
            else:
                self.update_checkpoint_ep(ep, model_path)
        self.update_checkpoint_finish(True, model_path)
        ckpt = torch.load(model_path)
        self.load_state_dict(ckpt['model'])
        elbo_te = self.ELBOX(te_dataset).item()
        return elbo_te


if __name__ == '__main__':
    pass
