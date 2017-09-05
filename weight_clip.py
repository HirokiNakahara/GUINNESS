import numpy
from chainer import cuda

class WeightClip(object):

    """Optimizer hook function for weight clip manipulation.

    This hook function clips a parameter to [low, high].
    It can be used in a binary weight network.

    Args:
        low (float): low value for the weight clip.
        high (float): high value for the weight clip.

    Attributes:
        low (float): low value for the weight clip.
        high (float): low value for the weight clip.

    """
    name = 'WeightClip'

    def __init__(self, low=-1.0, high=1.0):
        self.low=low
        self.high=high

    def __call__(self, opt):
        if cuda.available:
            kernel = cuda.elementwise(
                'T low, T high', 
                'T p', 
                'p = (p < low) ? low : (p > high) ? high : p',
                'weight_clip')

        for param in opt.target.params():
            p = param.data
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    numpy.clip(p, self.low, self.high)
                else:
                    kernel(self.low, self.high, p)
