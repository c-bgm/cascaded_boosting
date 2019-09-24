from boosting import *
from torch.utils.data import Dataset, DataLoader
from dataset import load_mnist_2d, QuickDataset
import torch.nn as nn
import torch.optim as optim
import sys
from MixGauss import MixGauss
import numpy as np

lr = 1e-3
batch_size = 256


class IWAE(MicroModule, nn.Module):
    def __init__(self, config):
        MicroModule.__init__(self, config)
        nn.Module.__init__(self)
        self.X_dim = int(np.prod(config['X_dim']))
        self.bn = False
        if "bn" in config and config['bn'] == True:
            self.bn = True

        self.Q_linear1 = nn.Linear(self.X_dim, config['h_dim'])
        self.Q_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.Qmu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.Qvar_linear = nn.Linear(config['h_dim'], config['Z_dim'])

        self.P_linear1 = nn.Linear(config['Z_dim'], config['h_dim'])
        self.P_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.Pmu_linear = nn.Linear(config['h_dim'], self.X_dim)

    def Q(self, X):
        h = F.relu(self.Q_linear1(X))
        h = F.relu(self.Q_linear2(h))
        Qmu = self.Qmu_linear(h)
        Qvar = F.softplus(self.Qvar_linear(h))
        return Qmu, Qvar

    def f(self, z):  # p(X|z) = N(X|f(z), sigma^2 I)
        h = F.relu(self.P_linear1(z))
        h = F.relu(self.P_linear2(h))
        X = F.sigmoid(self.Pmu_linear(h))
        return X

    def ELBOX__(self, X, tight=False): # ELBO of log p(x)
        return self.elbo(X)

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

    def sample_z(self, X, reduce_mean=True):
        Q_mu, Q_var = self.Q(X)
        eps = torch.randn(len(X), self.config['Z_dim']).cuda()
        z = Q_mu + eps * torch.sqrt(Q_var)
        logQ = self.logNormal(z, Q_mu, Q_var, reduce_mean=reduce_mean)
        return z, logQ

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
            Z, _ = self.sample_z(batch.cuda())
            Zs.append(Z.cpu().detach().numpy())
        Zs = np.concatenate(Zs)
        assert len(Zs) == len(X)
        return Zs

    def logNormal(self, X, mu, var, reduce_mean=True):
        X_dim = np.prod(list(X.shape[1:]))
        a = -0.5 * (X-mu) * (X-mu) / var
        a = torch.sum(a, dim=1)
        b = -0.5 * X_dim * np.log(2*np.pi)
        c = -0.5 * torch.log(var)
        c = torch.sum(c, dim=1)
        if reduce_mean:
            return torch.mean(a + b + c, dim=0)
        else:
            return a + b + c

    def logP(self, X, Z, reduce_mean=True):
        Z_mu = torch.zeros(*list(Z.shape)).cuda()
        Z_var = torch.ones(*list(Z.shape)).cuda()
        logPZ = self.logNormal(Z, Z_mu, Z_var, reduce_mean=reduce_mean)

        X_mu = self.f(Z)
        if reduce_mean:
            logPX = -F.binary_cross_entropy(X_mu, X, size_average=False) / len(X)
        else:
            logPX = -bernoulli_ce(X, X_mu)

        return logPZ + logPX

    def elbo(self, X, reduce_mean=True): # logp(x,z)-logq(z|x)
        z, logQ = self.sample_z(X, reduce_mean=reduce_mean)
        logP = self.logP(X, z, reduce_mean=reduce_mean)
        return logP - logQ

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
        z, _ = self.sample_z(X)

        # reconstruct
        samples = self.sampleX(z).cpu().data.numpy()
        samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
        savefig(samples, K, os.path.join(self.config['sample_dir'], 'reconstruct_{}.png'.format(str(ep).zfill(3))))

        # random sample
        z = torch.randn(100, self.config['Z_dim']).cuda()
        samples = self.sampleX(z).cpu().data.numpy()
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

    def cal_grad(self, X):
        param_grads = []
        weights = []
        param = list(self.parameters())
        for p in param:
            param_grads.append([])

        for i in range(self.config['IW']):
            elbo = self.elbo(X, reduce_mean=True)
            weights.append(elbo.item())
            loss = -elbo
            loss.backward()
            for p, grads in zip(param, param_grads):
                grads.append(p.grad.data.clone())
                p.grad.detach_()
                p.grad.zero_()
        weights = torch.Tensor(weights)
        weights = F.softmax(weights)
        for p, grads in zip(param, param_grads):
            temp = torch.zeros(*list(grads[0].shape)).cuda()
            for w, g in zip(weights, grads):
                temp = temp + w * g
            ctemp = temp.clone()
            p.grad.data = ctemp
        return elbo

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
                elbo = self.cal_grad(X)
                solver.step()
                solver.zero_grad()
                if 'disp_interval' in self.config and idx % self.config['disp_interval'] == 0:
                    print("[{}] [epoch {}/{}] [iter {}/{}] [elbo {}]"
                          .format(self.config['model_name'], ep, epoch,
                                  idx, len(tr_dataloader), elbo))
                    sys.stdout.flush()
            if save_fig:
                self.save_samples(te_dataset, ep)

            ep += 1
            # save model
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            elbo = self.ELBOX(val_dataset).item()
            elbo_te = self.ELBOX(te_dataset).item()
            print("[{}] [epoch {}/{}] [elbo_te: {:.2f}] [elbo {:.2f}] [max_elbo {:.2f}] [elbo_threshold {:.2f}]"
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


def experiment(id, phase, IW, binary=True):
    sample_path = 'exp_result/IWAE/exp%d' % id
    model_path = 'models/IWAE/exp%d' % id

    trX, teX, _, _ = load_mnist_2d('../models/data/mnist/raw')
    if binary:
        trX = binarize(trX)
        teX = binarize(teX)
    valX = trX[50000:]
    trX = trX[:50000]
    tr_dataset = QuickDataset(trX)
    val_dataset = QuickDataset(valX)
    te_dataset = QuickDataset(teX)
    K = 10

    if phase == "train":
        te_dataset = tr_dataset

    microConfigs = []
    microModules = []

    config1 = {'X_dim': (1, 28, 28), 'h_dim': 500, 'Z_dim': 50, 'IW': IW,
               'height': 28, 'width': 28, 'channel': 1,
               'lr': lr, 'bn': True,
               'epoch': 200, 'max_epoch': 200,
               'batch_size': batch_size,
               'sample_dir': os.path.join(sample_path, 'IWAE1'),
               'model_dir': os.path.join(model_path, 'IWAE1'),
               'model_name': 'IWAE1',
               'retrain': False,
               'phase': phase}

    microConfigs.append(config1)
    microModules.append(IWAE)

    config2 = {'X_dim': (50,), 'K': K,
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
    id = 0
    IWs = [5, 10]
    for IW in IWs:
        print("****************%d" % id)
        experiment(id=id, phase="train", IW=IW)
        experiment(id=id, phase="test", IW=IW)
        id += 1
