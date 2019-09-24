from boosting import *
from VAEGauss import VAEGaussLinear
from VAEBernoulli import VAEBernoulliLinear
from RBM import RBM
from dataset import load_mnist_2d, QuickDataset
import os

epoch = 50
max_epoch = 100


def experiment(combs, id, phase, binary=True):

    sample_path = 'exp_result/rbm_vae/exp%d' % id
    model_path = 'models/rbm_vae/exp%d' % id

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

    if combs[0] == "RBM":
        name = 'RBM1'
        config1 = RBM_config(X_dim=(28 * 28,), Z_dim=20,
                             height=28, width=28, channel=1,
                             epoch=epoch, max_epoch=max_epoch,
                             sample_dir=os.path.join(sample_path, name),
                             model_dir=os.path.join(model_path, name),
                             model_name=name,
                             retrain=False,
                             phase=phase)
        microConfigs.append(config1)
        microModules.append(RBM)
    else:
        name = 'VAEBernoulli1'
        config1 = VAEBernoulli_config(X_dim=(28 * 28,), h_dim=500, Z_dim=20,
                                      height=28, width=28, channel=1,
                                      sigma=0.5,
                                      epoch=epoch, max_epoch=max_epoch,
                                      sample_dir=os.path.join(sample_path, name),
                                      model_dir=os.path.join(model_path, name),
                                      model_name=name,
                                      retrain=False,
                                      phase=phase)
        config1["bn"] = True
        microConfigs.append(config1)
        microModules.append(VAEBernoulliLinear)

    for idx, name in enumerate(combs[1:]):
        if name == "RBM":
            name = 'RBM%d' % (idx + 2)
            config = RBM_config(X_dim=(20,), Z_dim=20,
                                height=4, width=4, channel=1,
                                epoch=epoch, max_epoch=max_epoch,
                                sample_dir=os.path.join(sample_path, name),
                                model_dir=os.path.join(model_path, name),
                                model_name=name,
                                retrain=False,
                                phase=phase)
            microConfigs.append(config)
            microModules.append(RBM)
        else:
            name = 'VAEGauss%d' % (idx + 2)
            config = VAEGauss_config(X_dim=(20,), h_dim=500, Z_dim=20,
                                     height=4, width=4, channel=1,
                                     sigma=0.5,
                                     epoch=epoch, max_epoch=max_epoch,
                                     sample_dir=os.path.join(sample_path, name),
                                     model_dir=os.path.join(model_path, name),
                                     model_name=name,
                                     retrain=False,
                                     phase=phase)
            config["bn"] = True
            microConfigs.append(config)
            microModules.append(VAEGaussLinear)

    config = {'microModules': microModules,
              'microConfigs': microConfigs,
              'dataset': (tr_dataset, val_dataset, te_dataset),
              'sample_dir': os.path.join(sample_path, 'boosting')}

    model = MacroModule(config)
    elbos, elbo_thresholds = model.build()
    display_elbo(elbos, elbo_thresholds)


if __name__ == "__main__":
    all_combs = [["VAE", "VAE", "VAE", "VAE", "VAE", "VAE"],
                 ["RBM", "RBM", "VAE", "VAE", "VAE", "VAE"],
                 ["RBM", "RBM", "RBM", "RBM", "VAE", "VAE"],
                 ["RBM", "RBM", "RBM", "RBM", "RBM", "RBM"]]
    for idx, comb in enumerate(all_combs):
        experiment(comb, idx, 'train')
        experiment(comb, idx, 'test')
