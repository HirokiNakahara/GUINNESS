import math
import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers

import sys
sys.path.append('./')
import link_binary_linear as BL
import bst
import link_binary_conv2d as BC
import link_integer_conv2d as IC
from function_binary_conv2d import func_convolution_2d
from function_integer_conv2d import func_convolution_2d

# for debuging of the batch normalization functions
import link_batch_normalization as LBN

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(

            conv0=IC.Convolution2D(3,64,3, stride=1, pad=1, nobias=True),
            b0=L.BatchNormalization(64),
            conv1=BC.Convolution2D(64,128,3, stride=1, pad=1, nobias=True),
            b1=L.BatchNormalization(128),
            conv2=BC.Convolution2D(128,128,3, stride=1, pad=1, nobias=True),
            b2=L.BatchNormalization(128),
            fc0=BL.BinaryLinear(128,3),
            b3=L.BatchNormalization(3)
        )

    def __call__(self, x, train):
        h = bst.bst(self.b0(self.conv0(x)))
        h = bst.bst(self.b1(self.conv1(h)))
        h = bst.bst(self.b2(self.conv2(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.average_pooling_2d(h, 24)
        h = self.b3(self.fc0(h))
        return h