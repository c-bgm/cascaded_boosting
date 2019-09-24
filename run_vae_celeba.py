from boosting import *
from VAEGauss import VAEGaussLinear
from VAEBernoulli import VAEBernoulliLinear
from dataset import CelebaFlatDataset
import os

epoch = 50
max_epoch = 100


def experiment(id, phase):

    sample_path = 'exp_result/vae_celeba/exp%d' % id
    model_path = 'models/vae_celeba/exp%d' % id

    dataset = CelebaFlatDataset("../models/data/celeba/resized_celeba")
    tr_dataset = dataset
    val_dataset = dataset
    te_dataset = dataset

    microConfigs = []
    microModules = []

    name = 'VAEBernoulli1'
    config1 = VAEBernoulli_config(X_dim=(3, 64, 64,), h_dim=2500, Z_dim=100,
                                  height=64, width=64, channel=3,
                                  sigma=0.5,
                                  epoch=epoch, max_epoch=max_epoch,
                                  sample_dir=os.path.join(sample_path, name),
                                  model_dir=os.path.join(model_path, name),
                                  model_name=name,
                                  retrain=False,
                                  phase=phase)
    config1["bn"] = True
    config1["disp_interval"] = 20
    microConfigs.append(config1)
    microModules.append(VAEBernoulliLinear)

    for i in range(2, 5):
        name = 'VAEGauss%d' % i
        config = VAEGauss_config(X_dim=(100,), h_dim=2500, Z_dim=100,
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
    experiment(0, 'train')
    experiment(0, 'test')
