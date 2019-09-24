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


class LVAE(MicroModule, nn.Module):
    def __init__(self, config):
        MicroModule.__init__(self, config)
        nn.Module.__init__(self)
        X_dim = int(np.prod(config['X_dim']))
        self.bn = False
        if "bn" in config and config['bn'] == True:
            self.bn = True
        self.d1_linear1 = nn.Linear(X_dim, config['h_dim'])
        self.d1_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.Q1_mu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.Q1_var_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.d2_linear1 = nn.Linear(config['h_dim'], config['h_dim'])
        self.d2_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.Q2_mu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.Q2_var_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        if self.bn:
            self.d1_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.d1_bn2 = nn.BatchNorm1d(config['h_dim'])
            self.d2_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.d2_bn2 = nn.BatchNorm1d(config['h_dim'])

        self.P1_linear1 = nn.Linear(config['Z_dim'], config['h_dim'])
        self.P1_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.P1_mu_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.P1_var_linear = nn.Linear(config['h_dim'], config['Z_dim'])
        self.P0_linear1 = nn.Linear(config['Z_dim'], config['h_dim'])
        self.P0_linear2 = nn.Linear(config['h_dim'], config['h_dim'])
        self.P0_mu_linear = nn.Linear(config['h_dim'], X_dim)
        if self.bn:
            self.P1_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.P1_bn2 = nn.BatchNorm1d(config['h_dim'])
            self.P0_bn1 = nn.BatchNorm1d(config['h_dim'])
            self.P0_bn2 = nn.BatchNorm1d(config['h_dim'])

    def d(self, X):
        if self.bn:
            d1 = F.relu(self.d1_bn1(self.d1_linear1(X)))
            d1 = F.relu(self.d1_bn2(self.d1_linear2(d1)))
            Q1_mu = self.Q1_mu_linear(d1)
            Q1_var = F.softplus(self.Q1_var_linear(d1))
            d2 = F.relu(self.d2_bn1(self.d2_linear1(d1)))
            d2 = F.relu(self.d2_bn2(self.d2_linear2(d2)))
            Q2_mu = self.Q2_mu_linear(d2)
            Q2_var = F.softplus(self.Q2_var_linear(d2))
        else:
            d1 = F.relu(self.d1_linear1(X))
            d1 = F.relu(self.d1_linear2(d1))
            Q1_mu = self.Q1_mu_linear(d1)
            Q1_var = F.softplus(self.Q1_var_linear(d1))
            d2 = F.relu(self.d2_linear1(d1))
            d2 = F.relu(self.d2_linear2(d2))
            Q2_mu = self.Q2_mu_linear(d2)
            Q2_var = F.softplus(self.Q2_var_linear(d2))
        return Q1_mu, Q1_var, Q2_mu, Q2_var

    def p(self, i, Z):
        if i == 1:
            if self.bn:
                t1 = F.relu(self.P1_bn1(self.P1_linear1(Z)))
                t1 = F.relu(self.P1_bn2(self.P1_linear2(t1)))
                P1_mu = self.P1_mu_linear(t1)
                P1_var = F.softplus(self.P1_var_linear(t1))
            else:
                t1 = F.relu(self.P1_linear1(Z))
                t1 = F.relu(self.P1_linear2(t1))
                P1_mu = self.P1_mu_linear(t1)
                P1_var = F.softplus(self.P1_var_linear(t1))
            return P1_mu, P1_var
        else:
            if self.bn:
                t0 = F.relu(self.P0_bn1(self.P0_linear1(Z)))
                t0 = F.relu(self.P0_bn2(self.P0_linear2(t0)))
                P0_mu = F.sigmoid(self.P0_mu_linear(t0))
            else:
                t0 = F.relu(self.P0_linear1(Z))
                t0 = F.relu(self.P0_linear2(t0))
                P0_mu = F.sigmoid(self.P0_mu_linear(t0))
            return P0_mu

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

    def sample_z(self, X):
        Q1_mu, Q1_var, Q2_mu, Q2_var = self.d(X)
        eps2 = torch.randn(len(X), self.config['Z_dim']).cuda()
        Z2 = Q2_mu + eps2 * torch.sqrt(Q2_var)
        logQZ2 = self.logNormal(Z2, Q2_mu, Q2_var)

        P1_mu, P1_var = self.p(1, Z2)

        Q1_var_ = 0.5 * (Q1_var + P1_var)
        Q1_mu_ = 0.5 * (Q1_mu + P1_mu)
        eps1 = torch.randn(len(X), self.config['Z_dim']).cuda()
        Z1 = Q1_mu_ + eps1 * torch.sqrt(Q1_var_)
        logQZ1 = self.logNormal(Z1, Q1_mu_, Q1_var_)

        return Z1, Z2, logQZ1, logQZ2

    def sampleX(self, Z=None):
        if Z is None:
            Z = torch.randn(self.config['batch_size'], self.config['Z_dim']).cuda()
        P1_mu, P1_var = self.p(1, Z)
        eps1 = torch.randn(len(Z), self.config['Z_dim']).cuda()
        Z1 = P1_mu + eps1 * torch.sqrt(P1_var)

        P0_mu = self.p(0, Z1)
        return P0_mu

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
            _, Z, _, _ = self.sample_z(batch.cuda())
            Zs.append(Z.cpu().detach().numpy())
        Zs = np.concatenate(Zs)
        assert len(Zs) == len(X)
        return Zs

    def logNormal(self, X, mu, var):
        X_dim = np.prod(list(X.shape[1:]))
        a = -0.5 * (X-mu) * (X-mu) / var
        a = torch.sum(a, dim=1)
        b = -0.5 * X_dim * np.log(2*np.pi)
        c = -0.5 * torch.log(var)
        c = torch.sum(c, dim=1)
        return torch.mean(a + b + c, dim=0)

    def logP(self, X, Z):
        Z1, Z2 = Z
        P2_mu = torch.zeros(*list(Z2.shape)).cuda()
        P2_var = torch.ones(*list(Z2.shape)).cuda()
        logPZ2 = self.logNormal(Z2, P2_mu, P2_var)

        P1_mu, P1_var = self.p(1, Z2)
        logPZ1 = self.logNormal(Z1, P1_mu, P1_var)

        P0_mu = self.p(0, Z1)
        logPZ0 = -F.binary_cross_entropy(P0_mu, X, size_average=False) / len(X)

        return logPZ2 + logPZ1 + logPZ0

    def elbo(self, X):
        Z1, Z2, logQZ1, logQZ2 = self.sample_z(X)
        logQ = logQZ1 + logQZ2
        logP = self.logP(X, (Z1, Z2))
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
        _, z, _, _ = self.sample_z(X)

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
                elbo = self.elbo(X)
                loss = -elbo
                loss.backward()
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


def experiment(id, phase, binary=True):
    sample_path = 'exp_result/LVAE/exp%d' % id
    model_path = 'models/LVAE/exp%d' % id

    trX, teX, _, _ = load_mnist_2d('../models/data/mnist/raw')
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

    config1 = {'X_dim': (1, 28, 28), 'h_dim': 1000, 'Z_dim': 30,
               'sigma': 1.0,
               'height': 28, 'width': 28, 'channel': 1,
               'lr': lr, 'bn': True,
               'epoch': 200, 'max_epoch': 200,
               'batch_size': batch_size,
               'sample_dir': os.path.join(sample_path, 'LVAE1'),
               'model_dir': os.path.join(model_path, 'LVAE1'),
               'model_name': 'LVAE1',
               'retrain': False,
               'phase': phase}
    microConfigs.append(config1)
    microModules.append(LVAE)

    config2 = {'X_dim': (30,), 'K': 10,
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
