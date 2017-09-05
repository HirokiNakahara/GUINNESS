import numpy as np
import six
from scipy.misc import imresize, imrotate

from chainer import functions as F
from chainer import cuda
from chainer import Variable

class CifarTrainer(object):
    def __init__(self, net, optimizer, epoch_num=100, batch_size=100, device_id=-1):
        self.net = net
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device_id = device_id
        if device_id >= 0:
            self.xp = cuda.cupy
            self.net.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, x, y, valid_x, valid_y, img_siz, img_dim, test_x=None, test_y=None, callback=None):
        if self.device_id >= 0:
            with cuda.cupy.cuda.Device(self.device_id):
                return self.__fit(x, y, valid_x, valid_y, img_siz, img_dim, test_x, test_y, callback)
        else:
            return self.__fit(x, y, valid_x, valid_y, img_siz, img_dim, test_x, test_y, callback)

    def __fit(self, x, y, valid_x, valid_y, img_siz, img_dim, test_x, test_y, callback):
        batch_size = self.batch_size
        for epoch in six.moves.range(self.epoch_num):
            perm = np.random.permutation(len(x))
            train_loss = 0
            train_acc = 0
            for i in six.moves.range(0, len(x), self.batch_size):
                self.net.zerograds()
                batch_index = perm[i:i + batch_size]
                x_batch = self.__trans_image(x[batch_index], img_siz, img_dim)

#                print(type(x_batch[0,0,0,0]))
#                print(batch_index)
#                print(type(y[batch_index][0]))

                loss, acc = self.__forward(x_batch, y[batch_index])
                loss.backward()
                self.optimizer.update()
                train_loss += float(loss.data) * len(x_batch)
                train_acc += float(acc.data) * len(x_batch)
            train_loss /= len(x)
            train_acc /= len(x)
            valid_loss = 0
            valid_acc = 0
            if valid_x is not None and valid_y is not None:
                for i in six.moves.range(0, len(valid_x), self.batch_size):
                    x_batch = valid_x[i:i + batch_size]
                    loss, acc = self.__forward(x_batch, valid_y[i:i + batch_size], train=False)
                    valid_loss += float(loss.data) * len(x_batch)
                    valid_acc += float(acc.data) * len(x_batch)
            valid_loss /= len(valid_x)
            valid_acc /= len(valid_x)
            test_loss = 0
            test_acc = 0
            if test_x is not None and test_y is not None:
                for i in six.moves.range(0, len(test_x), self.batch_size):
                    x_batch = test_x[i:i + batch_size]
                    loss, acc = self.__forward(x_batch, test_y[i:i + batch_size], train=False)
                    test_loss += float(loss.data) * len(x_batch)
                    test_acc += float(acc.data) * len(x_batch)
                test_loss /= len(test_x)
                test_acc /= len(test_x)
            if callback is not None:
                callback(epoch, self.net, self.optimizer, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)

    def __forward(self, batch_x, batch_t, train=True):
        xp = self.xp
        x = Variable(xp.asarray(batch_x), volatile=not train)
        t = Variable(xp.asarray(batch_t), volatile=not train)
        y = self.net(x, train=train)
#        print(type(y.data))
#        print(type(t.data))
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        return loss, acc

    def __trans_image(self, x, img_siz, img_dim):
        size = img_siz
        n = x.shape[0]
        images = np.zeros((n, img_dim, size, size), dtype=np.float32)
        offset = np.random.randint(-4, 5, size=(n, 2))
        mirror = np.random.randint(2, size=n)
        for i in six.moves.range(n):
            image = x[i]
            top, left = offset[i]
            left = max(0, left)
            top = max(0, top)
            right = min(size, left + size)
            bottom = min(size, left + size)
            if mirror[i] > 0:
                images[i,:,size-bottom:size-top,size-right:size-left] = image[:,top:bottom, left:right][:,:,::-1]
            else:
                images[i,:,size-bottom:size-top,size-right:size-left] = image[:,top:bottom,left:right]
        return images
