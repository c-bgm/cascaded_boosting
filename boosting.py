import abc
from dataset import QuickDataset, LabelledDataset
import os
import torch
from utils import *
import random


class MicroModule(object):
    def __init__(self, config):
        self.config = config
        pass

    @abc.abstractmethod
    def build(self, data):  # train or load model parameter
        pass

    @abc.abstractmethod
    def sampleX(self, Z=None):  # samples from p(X|Z) or p(X)
        pass

    @abc.abstractmethod
    def sampleZ(self, X=None):  # samples from q(Z|X)
        return None

    def plot2dX(self):  # plot samples from p(X)
        samples = []
        for i in range(500):
            samples.append(self.sampleX().cpu().detach().numpy())
        samples = np.concatenate(samples, axis=0)
        plot2d(samples)

    def plot2dZ(self, X):  # plot samples from q(Z|X)
        samples = self.sampleZ(X)
        plot2d(samples)


class MacroModule(object):
    def __init__(self, config):
        self.config = config
        self.microModuleList = []

    def build(self):
        dataset = self.config['dataset']
        elbo_threshold = -10.e10
        elbos = []
        elbo_thresholds = []
        for i, (micro, config) in enumerate(zip(self.config['microModules'], self.config['microConfigs'])):
            model = micro(config)
            elbo = model.build(dataset, elbo_threshold=elbo_threshold)
            elbos.append(elbo)
            self.microModuleList.append(model)
            self.sample(i + 1)
            if i < len(self.config['microModules']) - 1:
                tr, val, te = dataset  # in train phase, te = tr
                tr_data = model.sampleZ(tr)
                val_data = model.sampleZ(val)
                te_data = model.sampleZ(te)
                tr_dataset = QuickDataset(tr_data)
                val_dataset = QuickDataset(val_data)
                te_dataset = QuickDataset(te_data)
                dataset = (tr_dataset, val_dataset, te_dataset)
                elbo_threshold = model.ELBOZ(te_dataset)
                elbo_thresholds.append(elbo_threshold)
        return elbos, elbo_thresholds

    def sample(self, num_modules, sample_reconstruct=False):
        if not os.path.exists(self.config['sample_dir']):
            os.makedirs(self.config['sample_dir'])
        config = self.microModuleList[0].config
        height, width, channel = config['height'], config['width'], config['channel']

        # save random samples
        sample_batch_size = self.microModuleList[num_modules-1].config['batch_size']
        sample_times = int(100./sample_batch_size) + 1
        samples = []
        for _ in range(sample_times):
            cond = None
            for k, model in enumerate(self.microModuleList[:num_modules][::-1]):
                cond = model.sampleX(cond)
            samples.append(cond.cpu().data.numpy())
        samples = np.concatenate(samples, axis=0)
        samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
        savefig(samples, 10, os.path.join(self.config['sample_dir'], 'num_module_%d.png' % num_modules))

        if sample_reconstruct:
            # save original samples
            origin = []
            dataset = self.config['dataset'][0]
            idxes = list(range(len(dataset)))
            random.shuffle(idxes)
            for i in idxes[:100]:
                origin.append(dataset[i])
            origin = np.array(origin)
            K = 10
            if len(origin) < 100:
                K = int(len(origin) ** 0.5)
            samples = arr2img(np.clip(origin, 0., 1.), height, width, channel)
            savefig(samples, K, os.path.join(self.config['sample_dir'], 'origin_num_module_%d.png' % num_modules))

            # save reconstruct samples
            X = origin
            cond = self.sample_q(X, num_modules)
            for k, model in enumerate(self.microModuleList[:num_modules][::-1]):
                cond = model.sampleX(cond)
            samples = cond.cpu().data.numpy()
            samples = arr2img(np.clip(samples, 0., 1.), height, width, channel)
            savefig(samples, K, os.path.join(self.config['sample_dir'], 'reconstruct_num_module_%d.png' % num_modules))

    def cal_ELBO(self, X, tight=False):  # X is a data point
        k = len(self.microModuleList)
        X = torch.Tensor([X]).cuda()
        ELBOs = []
        ELBO_incs = []
        model = self.microModuleList[0] # m1
        ELBO = model.ELBOX__(X).cpu().detach().numpy()
        # print("[ELBO(m_1(x)) %f] " % ELBO)
        ELBOs.append(ELBO)
        i = 1
        while i < k:
            model = self.microModuleList[i]
            oldmodel = self.microModuleList[i-1]
            X = oldmodel.sampleZ(X.cpu().detach().numpy())
            X = torch.Tensor(X).cuda()
            # dataset = QuickDataset(data)
            oldmodel_ELBO = oldmodel.ELBOZ__(X).cpu().detach().numpy()
            # print("[ELBO(m_%d(h_%d)) %f]" % (i, i, oldmodel_ELBO))
            model_ELBO = model.ELBOX__(X, tight).cpu().detach().numpy()
            # print("[ELBO(m_%d(h_%d)) %f] " % (i+1, i, model_ELBO))
            ELBO_inc = model_ELBO - oldmodel_ELBO
            ELBO = ELBO + ELBO_inc
            ELBO_incs.append(ELBO_inc)
            ELBOs.append(ELBO)
            i += 1
        return ELBOs[-1]

    def sample_q(self, X, k):  # samples from q(z_k|X)
        assert k >= 1
        Z = X
        for i, model in enumerate(self.microModuleList[:k]):
            Z = model.sampleZ(Z)
        return Z

    def sample_p(self, k):  # sample from p(z_k)
        assert k >= 0
        n = len(self.microModuleList)
        Z = self.microModuleList[n-1].sampleZ()  # z_n
        Z = torch.from_numpy(Z).float().cuda()
        while n > k:
            Z = self.microModuleList[n-1].sampleX(Z)  # z_n-1
            n -= 1
        return Z

    def plot2d_q(self, X, k):  # plot samples from q(z_k|X); z_0 = X
        Z = self.sample_q(X, k)
        plot2d(Z)

    def plot2d_p(self, k):  # plot sample from p(z_k)
        samples = []
        for i in range(500):
            samples.append(self.sample_p(k).cpu().detach().numpy())
        samples = np.concatenate(samples, axis=0)
        plot2d(samples)
