import tensorflow as tf
from tensorflow import keras

import numpy as np


class Node(object):
    def __init__(self,id,depth,pathprob,tree):
        self.id = id
        self.depth = depth
        self.prune(tree)
        if self.isLeaf:
            self.W = tf.get_variable(...)
            self.b = tf.get_variable(...)
        else:
            self.W = tf.get_variable(...)
            self.b = tf.get_variable(...)

        self.leftChild = None
        self.rightChild = None

        self.pathprob = pathprob
        self.epsilon = 1e-8 #this is a correction to avoid log(0)

    def prune(self,tree):
        '''
        prunes the leaf by setting isLeaf to True if the pruning condition applies.
        :param tree:
        '''
        self.isLeaf = (self.depth>=tree.params.max_depth)
    def build(self,x,tree):
        '''
        define the output probability of the node and build the children
        :param x:
        :return:
        '''
        self.prob = self.forward(x)

        if not(self.isLeaf):
            self.leftChild = Node(...,pathprob=self.pathprob * self.prob)
            self.rightChild = Node(..., pathprob=self.pathprob * (1. - self.prob))

    def forward(self,x):
        '''
        defines the output probability
        :param x:
        :return:
        '''
        if self.isLeaf:
            return tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        else:
            return tf.keras.backend.hard_sigmoid(tf.matmul(x, self.W) + self.b)

    def regularise(self,tree):
        if self.isLeaf:
            return 0.0
        else:
            alpha = tf.reduce_mean(self.pathprob * self.prob) / (
                        self.epsilon + tf.reduce_mean(self.pathprob))
            return (-0.5 * tf.log(alpha + self.epsilon) - 0.5 * tf.log(
                1. - alpha + self.epsilon)) * (tree.params.decay_penality** self.depth)

    def get_loss(self,y,tree):
        if self.isLeaf:
            return -tf.reduce_mean( tf.log( self.epsilon+tf.reduce_sum(y *self.prob, axis=1) )*self.pathprob  )
        else:
            return tree.params.regularisation_penality * self.regularise(tree)