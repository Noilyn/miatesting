import os
import sys
import time
import torch
import torch.nn as nn
import random
import numpy as np
import abc

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, '../../data')
from .misc import Partition, get_all_losses, savefig
from .logger import AverageMeter, Logger
from.eval import accuracy, accuracy_binary
from progress.bar import Bar as Bar

__all__ = ['BaseTrainer']


class BaseTrainer(object):
    """The class that contains the code for base trainer class."""

    def __init__(self, the_args, save_dir):
        """The function to initialize this class"""
        self.args = the_args
        self.save_dir = save_dir
        self.data_root = DATA_ROOT
        self.set_cuda_device()
        self.set_seed()
        self.set_dataloader()
        self.set_logger()
        self.set_criterion()

    def set_cuda_device(self):
