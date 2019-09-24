from boosting import *
from dataset import QuickDataset, GaussDataset
from utils import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import mixture
import time


class MixGauss(MicroModule):
    def __init__(self, config):
        MicroModule.__init__(self, config)
        self.K = config['K']
        self.X_dim = int(np.prod(config['X_dim']))
        self.clf = mixture.GaussianMixture(n_components=self.K, covariance_type='full')

    def ELBOX__(self, X, tight=False): # logpX
        XX = X.detach().cpu().numpy()
        res = self.clf.score_samples(XX)
        return np.mean(res, axis=0)

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
            ELBO = self.ELBOX__(batch.cuda(), tight=tight)
            ELBOs.append(ELBO)
        return np.mean(ELBOs)

    def sampleX(self, Z=None):
        Xs = self.clf.sample(self.config['batch_size'])[0]
        np.random.shuffle(Xs)
        return torch.from_numpy(Xs).float().cuda()

    def save_samples(self, dataset, ep):
        return None

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

    def build(self, dataset, elbo_threshold=-10.0e10, save_fig=False):  # train model or load model parameters
        model_path = os.path.join(self.config['model_dir'], '%s.pth' % self.config['model_name'])
        tr_dataset, val_dataset, te_dataset = dataset
        tr_arr = dataset2Arr(tr_dataset)
        val_arr = dataset2Arr(val_dataset)
        te_arr = dataset2Arr(te_dataset)
        tr = np.concatenate([tr_arr, val_arr], axis=0)

        st = time.time()
        self.clf.fit(tr)
        ed = time.time()
        print("[extra time %.2f]" % (ed - st))

        elbo_te = np.mean(self.clf.score_samples(te_arr), axis=0)
        return elbo_te


def experiment(id):
    sample_path = 'exp_result/MixGauss/exp%d' % id
    model_path = 'models/MixGauss/exp%d' % id
    dim = 10
    mu = np.array(list(range(dim)), dtype=np.float32)
    var = np.ones([dim], dtype=np.float32)
    tr_dataset = GaussDataset(mu, var, dim)
    val_dataset = GaussDataset(mu, var, dim)
    te_dataset = GaussDataset(mu, var, dim)
    dataset = (tr_dataset, val_dataset, te_dataset)

    config = {'X_dim': (dim, ), 'K': 10,
               'batch_size': 256,
               'sample_dir': os.path.join(sample_path, 'MixGauss'),
               'model_dir': os.path.join(model_path, 'MixGauss'),
               'model_name': 'MixGauss', 'phase': 'train',
               'retrain': False}

    model = MixGauss(config)
    print(model.build(dataset))


if __name__ == "__main__":
    experiment(0)
