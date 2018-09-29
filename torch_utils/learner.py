import torch
import numpy as np

import torch_funcs as thf
import metrics
import dataset as ds

import matplotlib.pyplot as plt
from tqdm import tqdm


class Learner:
    def __init__(self, model, loss, opt, metrics=[]):
        self.model = model
        self.loss = loss
        self.opt = opt
        if self.opt is not None:
            for group in self.opt.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.metrics = metrics

    def make_step(self, data, training=False):
        image = torch.autograd.Variable(data['image']).cuda()
        mask = torch.autograd.Variable(data['mask']).cuda()

        prediction = self.model(image)
        losses = { 'loss': self.loss(prediction, mask) }
        prediction = torch.sigmoid(
            prediction,
        ).data.cpu().numpy()

        if training:
            losses['loss'].backward()
            self.opt.step()

        mask = mask.data.cpu().numpy()
        for metric in self.metrics:
            losses.update(metric(prediction, mask))

        return losses
        
    def train_on_epoch(self, datagen, hard_negative_miner=None, lr_scheduler=None):
        self.model.train()
        meters = list()

        for data in tqdm(datagen):
            self.opt.zero_grad()

            meters.append(self.make_step(data, training=True))
            if lr_scheduler is not None:
                if hasattr(lr_scheduler, 'batch_step'):
                    lr_scheduler.batch_step()

            if hard_negative_miner is not None:
                hard_negative_miner.update_cache(meters[-1], data)
                if hard_negative_miner.need_iter():
                    self.make_step(hard_negative_miner.get_cache(), training=True)
                    hard_negative_miner.invalidate_cache()

        self.opt.zero_grad()
        return metrics.aggregate(meters)

    def validate(self, datagen):
        self.model.eval()
        meters = list()

        with torch.no_grad():
            for data in tqdm(datagen):
                meters.append(self.make_step(data, training=False))

        return metrics.aggregate(meters)

    def infer_on_data(self, data, verbose=True):
        if self.model.training:
            self.model.eval()
        if len(data['image'].shape) == 3:
            data['image'] = data['image'].unsqueeze(0)

        image = torch.autograd.Variable(data['image']).cuda()
        pred = self.model.forward(image)
        pred = torch.sigmoid(pred).data.cpu().numpy()
        if verbose:
            image = np.rollaxis(data['image'][0].numpy(), 0, 3)
            image = (image * ds.STD + ds.MEAN)

            _, ax = plt.subplots(ncols=3, figsize=(15, 5))
            ax[0].imshow(np.squeeze(image))
            ax[1].imshow(np.squeeze(data['mask'].numpy()))
            ax[2].imshow(np.squeeze(pred[0]))
            plt.show()

        return pred

    def freeze_encoder(self, unfreeze=False):
        if hasattr(self.model, 'module'):
            encoder = self.model.module.encoder
        elif hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        thf.freeze(encoder, unfreeze=unfreeze)
        thf.unfreeze_bn(encoder)
