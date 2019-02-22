import numpy as np


class LRFinder:
    def __init__(
        self, optimizer, 
        min_lr=1e-5, max_lr=1e-2, 
        steps_per_epoch=None, epochs=None
    ):

        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

        self.batch_step(self.iteration)


    def get_lr(self):
        '''Calculate the learning rate.'''
        x = self.iteration % self.total_iterations
        lr = self.max_lr - (self.max_lr - self.min_lr) * x

        lrs = list()
        for param_group in self.optimizer.param_groups:
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None, logs=None):
        self.iteration = batch_iteration or self.iteration + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if logs is not None:
            self.history.setdefault('lr', []).append(lr)
            self.history.setdefault('iterations', []).append(self.iteration)

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')


def find_lr(model, datagen_params, min_lr=1e-5, max_lr=1e-2, epochs=3):
    train_datagen, val_datagen = get_datagens(max_negatives=2000, **datagen_params)
    opt = torch.optim.SGD(model.parameters(), lr=min_lr, momentum=.9, weight_decay=1e-4)
    learner = RetinaLearner(model=model, opt=opt, loss=None, clf_loss=None, metrics=[], clf_reg_alpha=.5, ignored_keys=['clf_out'])

    print('steps per epoch: {}'.format(len(train_datagen)))
    lr_scheduler = LRFinder(learner.opt, min_lr=min_lr, max_lr=max_lr, steps_per_epoch=len(train_datagen), epochs=epochs)
    learner, history = orchestrate(
        learner=learner, train_datagen=train_datagen, val_datagen=val_datagen, epochs=epochs,
        lr_scheduler=lr_scheduler, checkpoints_pth=None, nb_freezed_epchs=-1, df=datagen_params['df'],
    )
    return learner, lr_scheduler
