import tensorflow as tf
from tensorflow import keras

import numpy as np


class Tree(object):
    '''
    :param max_leafs: maximum number of leafs
    :param n_features: maximum number of feature available within the data
    :param n_classes: number of classes
    '''
    def __init__(self,max_depth,max_leafs,n_features,n_classes,regularisation_penality=10.,decay_penality=0.9):
        self.max_depth = max_depth
        self.max_leafs = max_leafs
        self.n_features = n_features
        self.n_classes = n_classes

        self.epsilon = 1e-8

        self.decay_penality = decay_penality
        self.regularisation_penality = regularisation_penality