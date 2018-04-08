import math
import os
import glob

from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback


class Snapshot(Callback):

    def __init__(self, folder_path, nb_epochs, verbose=0, nb_cycles=5):
        if nb_cycles > nb_epochs:
            raise ValueError('nb_epochs has to be lower than nb_cycles.')

        super(Snapshot, self).__init__()
        self.verbose = verbose
        self.folder_path = folder_path
        self.nb_epochs = nb_epochs
        self.nb_cycles = nb_cycles
        self.period = self.nb_epochs // self.nb_cycles
        self.nb_digits = len(str(self.nb_cycles))
        self.path_format = os.path.join(self.folder_path, 'weights_cycle_{}.h5')


    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch % self.period != 0: return
        # Only save at the end of a cycle, a not at the beginning

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        cycle = int(epoch / self.period)
        cycle_str = str(cycle).rjust(self.nb_digits, '0')
        self.model.save_weights(self.path_format.format(cycle_str), overwrite=True)

        # Resetting the learning rate
        K.set_value(self.model.optimizer.lr, self.base_lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Reached %d-th cycle, saving model.' % (epoch, cycle))


    def on_epoch_begin(self, epoch, logs=None):
        if epoch <= 0: return

        lr = self.schedule(epoch)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Snapchot modifying learning '
                  'rate to %s.' % (epoch + 1, lr))


    def set_model(self, model):
        self.model = model
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get initial learning rate
        self.base_lr = float(K.get_value(self.model.optimizer.lr))


    def schedule(self, epoch):
        lr = math.pi * (epoch % self.period) / self.period
        lr = self.base_lr / 2 * (math.cos(lr) + 1)
        return lr

