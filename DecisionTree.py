import tensorflow as tf
from tensorflow import keras

import numpy as np

import Tree
import Node

class DecisionTree(object):
    def __init__(self, *args,**kwargs):
        self.params = Tree(*args,**kwargs)

        self.loss = 0.0

        self.output = list()
        self.leafs_distribution = list()
    def build_tree(self):
        self.tf_X = tf.placeholder(tf.float32, [None, self.params.n_features])
        self.tf_y = tf.placeholder(tf.float32, [None, self.params.n_classes])
        leafs = list()
        self.root = Node(...,pathprob=tf.constant(1.0,shape=(1,)))
        leafs.append(self.root )

        for node in leafs:
            self.n_nodes+=1
            node.build(x=self.tf_X,tree=self)
            self.loss += node.get_loss(y=self.tf_y, tree=self)

            self.add_node()
            self.add_leaf(node)
            if node.isLeaf:
                self.output.append(node.prob)
                self.leafs_distribution.append(node.pathprob)
            else:
                leafs.append(node.leftChild)
                leafs.append(node.rightChild)

        self.output = tf.concat(self.output,axis=1)
        self.leafs_distribution = tf.concat(self.leafs_distribution,axis=1)