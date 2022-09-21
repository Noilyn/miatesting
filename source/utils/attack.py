import os
import numpy as np
import math
import random
import time
import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.stats import norm, kurtosis, skew
from sklearn import metrics
from tqdm import tqdm
from progress.bar import Bar as Bar
from .logger import Logger, AverageMeter
from .eval import accuracy_binary, accuracy, metrics_binary
from .misc import *
from .base import BaseTrainer


__all__ = ['Benchmark', 'Benchmark_Blackbox', 'BaseAttacker']


class Benchmark(object):
    def __init__(self, shadow_train_scores, shadow_test_scores, target_train_scores,
                 target_test_scores):
        self.s_tr_scores = shadow_train_scores
        self.s_te_scores = shadow_test_scores
        self.t_tr_scores = target_train_scores
        self.t_te_scores = target_test_scores
        self.num_methods = len(self.s_tr_scores)

    def load_labels(self, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes):
        """Load sample labels"""
        self.num_classes = num_classes
        self.s_tr_labels = s_tr_labels
        self.s_te_labels = s_te_labels
        self.t_tr_labels = t_tr_labels
        self.t_te_labels = t_te_labels

    def _thre_setting(self, tr_values, te_values):
        """Set the best threshold"""
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre_perclass(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA by thresholding per-class feature values """
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.numclasses):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num], s_te_values[
                self.s_te_labels == num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(
            self.t_te_labels) + 0.0))
        info = 'MIA via {n} (pre-class threshold): the attack acc is {acc:.3f}'.format(n=v_name,
            acc=mem_inf_acc)
        print(info)
        return info, mem_inf_acc