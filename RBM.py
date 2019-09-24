from boosting import MicroModule
from dataset import QuickDataset
from utils import *
import numpy as np
from dataset import load_mnist_2d
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import os


class RBM(MicroModule):
    def __init__(self, config):
        MicroModule.__init__(self, config)
        self.X_dim = int(np.prod(self.config['X_dim']))
        self.W = torch.Tensor(np.random.randn(self.X_dim, self.config['Z_dim']) * 0.001).cuda()
        self.a = torch.Tensor(np.random.randn(self.config['Z_dim']) * 0.001).cuda()
        self.b = torch.Tensor(np.random.randn(self.X_dim) * 0.001).cuda()
        self.initX = None
        self.logZ = None
        self.sample_times = 20
        if 'sample_times' in config.keys():
            self.sample_times = config['sample_times']

    def energy(self, X, Z):
        if X.dtype is torch.float64:
            return - (X @ self.b.double() + Z @ self.a.double() + dot_elementwise(X @ self.W.double(), Z))
        else:
            return - (X @ self.b + Z @ self.a + dot_elementwise(X @ self.W, Z))

    def cal_logZ(self):
        pZ = torch.ones(self.config['batch_size'], self.config['Z_dim']).cuda() / 2.
        C = np.log(2) * self.config['Z_dim']
        C = C.item()
        num_iters = int(20000. / self.config['batch_size'])
        res_sum = 0.
        for k in range(num_iters):
            Z = torch.bernoulli(pZ)
            u = self.f(Z)
            X = torch.bernoulli(u)
            CE = bernoulli_ce(X, u).double()
            E = self.energy(X, Z).double()
            expE = torch.exp(C + CE - E)
            res_sum += torch.mean(expE, dim=0)
        res_sum /= num_iters
        res_sum = torch.log(res_sum)
        res_sum = res_sum.float()
        self.logZ = res_sum
        return self.logZ

    def ELBOX__(self, X, tight=False):  # ELBO of log p(x)
        if self.logZ is None:
            self.cal_logZ()
        z = self.Q(X)
        energy = torch.mean(self.energy(X, z), dim=0)
        H = torch.mean(bernoulli_entropy(z), dim=0)
        res = -energy + H - self.logZ
        return res

    def ELBOZ__(self, Z): # ELBO of log p(z)
        if self.logZ is None:
            self.cal_logZ()
        X = self.f(Z)
        energy = torch.mean(self.energy(X, Z), dim=0)
        H = torch.mean(bernoulli_entropy(X), dim=0)
        res = -energy + H - self.logZ
        return res

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
        return np.mean(ELBOs, axis=0)

    def ELBOZ(self, Z):
        if isinstance(Z, np.ndarray):
            dataset = QuickDataset(Z)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        elif isinstance(Z, Dataset):
            dataloader = DataLoader(Z, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        else:
            print("[ELBO] [Z type % s]" % type(Z))
        ELBOs = []
        for batch in dataloader:
            ELBO = self.ELBOZ__(batch.cuda()).cpu().detach().numpy()
            ELBOs.append(ELBO)
        return np.mean(ELBOs, axis=0)

    def f(self, z):  # p(X|z)
        return torch.sigmoid(z @ self.W.t() + self.b)
        # return sigmoid(z @ self.W.T + self.b)

    def Q(self, X):  # Q(z|X)
        return torch.sigmoid(X @ self.W + self.a)
        # return sigmoid(X @ self.W + self.a)

    def sampleX(self, Z=None):
        if Z is None:
            X = self.initX
            for i in range(self.sample_times):
                Z = torch.bernoulli(self.Q(X))
                X = torch.bernoulli(self.f(Z))
            Z = torch.bernoulli(self.Q(X))
        X = self.f(Z)
        return X

    def sampleZ(self, X=None):
        if X is None:
            print("[sampleZ] [X is None]")
            exit(-1)
        elif isinstance(X, np.ndarray):
            dataset = QuickDataset(X)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        elif isinstance(X, Dataset):
            dataloader = DataLoader(X, batch_size=self.config['batch_size'], shuffle=False)
        else:
            print("[sampleZ] [X type % s]" % (type(X)))
            exit(-1)
        Zs = []
        for batch in dataloader:
            Z = self.Q(batch.cuda())
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

        samples = arr2img(np.clip(origin, 0., 1.), height, width, channel)
        savefig(samples, 10, os.path.join(self.config['sample_dir'], 'origin_{}.png'.format(str(ep).zfill(3))))

        # X = origin
        X = torch.from_numpy(origin).float().cuda()
        # z = np.random.binomial(n=1, p=self.Q(X))
        z = torch.bernoulli(self.Q(X))

        # reconstruct
        samples = self.f(z).cpu().detach().numpy()
        samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
        savefig(samples, 10, os.path.join(self.config['sample_dir'], 'reconstruct_{}.png'.format(str(ep).zfill(3))))

        # random sample
        samples = []
        times = int(100. / self.config['batch_size']) + 1
        for _ in range(times):
            samples.append(self.sampleX().cpu().detach().numpy())
        samples = np.concatenate(samples, axis=0)
        samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
        savefig(samples, 10, os.path.join(self.config['sample_dir'], '{}.png'.format(str(ep).zfill(3))))

    def save_checkpoint(self, ep, elbo, W, a, b, path):
        ckpt = {
            'ep': ep,
            'elbo': elbo,
            'W': W,
            'a': a,
            'b': b,
            'finish': False
        }
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    def load_ckpt(self, ckpt):
        self.W = ckpt['W']
        self.a = ckpt['a']
        self.b = ckpt['b']

    def updata_checkpoint_ep(self, ep, path):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        ckpt['ep'] = ep
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    def updata_checkpoint_finish(self, flag, path):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        ckpt['finish'] = True
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    def build(self, dataset, elbo_threshold=-10.0e10, save_fig=True):  # train model or load model parameters
        model_path = os.path.join(self.config['model_dir'], '%s.pth' % self.config['model_name'])
        tr_dataset, val_dataset, te_dataset = dataset
        tr_dataloader = DataLoader(tr_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        self.initX = next(iter(tr_dataloader)).cuda()

        ep = 0
        epoch = self.config['epoch']
        max_elbo = float("-inf")
        finish = False
        if not self.config['retrain'] and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                ckpt = pickle.load(f)
            self.load_ckpt(ckpt)
            ep = ckpt['ep']
            max_elbo = ckpt['elbo']
            finish = ckpt['finish']

        if self.config['phase'] != "train":
            elbo = self.ELBOX(te_dataset).item()
            return elbo

        while not finish and ep < self.config['max_epoch'] and not (ep >= epoch and max_elbo > elbo_threshold):
            for idx, batch in enumerate(tr_dataloader):
                X = batch.cuda()

                h = self.Q(X)
                X_prime = X
                for k in range(self.sample_times):
                    h_prime = torch.bernoulli(self.Q(X_prime))
                    X_prime = torch.bernoulli(self.f(h_prime))

                h_prime = self.Q(X_prime)
                grad_W = torch.mean(outer(X, h), dim=0) - torch.mean(outer(X_prime, h_prime), dim=0)
                grad_a = torch.mean(h, dim=0) - torch.mean(h_prime, dim=0)
                grad_b = torch.mean(X, dim=0) - torch.mean(X_prime, dim=0)

                self.W += self.config['lr'] * grad_W
                self.a += self.config['lr'] * grad_a
                self.b += self.config['lr'] * grad_b

            if save_fig:
                self.save_samples(tr_dataset, ep)

            ep += 1
            # save model
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            elbo = self.ELBOX(val_dataset).item()
            elbo_te = self.ELBOX(te_dataset).item()
            self.logZ = None
            print("[{}] [epoch {}/{}] [elbo_te {:.2f}] [elbo_val {:.2f}] [max_elbo {:.2f}] [elbo_threshold {:.2f}]"
                  .format(self.config['model_name'], ep, epoch, elbo_te, elbo, max_elbo, elbo_threshold))
            if elbo >= max_elbo:
                max_elbo = elbo
                self.save_checkpoint(ep, elbo, self.W, self.a, self.b, model_path)
            else:
                self.updata_checkpoint_ep(ep, model_path)
        self.updata_checkpoint_finish(True, model_path)
        with open(model_path, "rb") as f:
            ckpt = pickle.load(f)
        self.load_ckpt(ckpt)
        elbo_te = self.ELBOX(te_dataset).item()
        return elbo_te


if __name__ == '__main__':
    pass
