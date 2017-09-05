import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

def _kern():
    return cuda.elementwise(
        'T x', 'T y',
        'y = x >= 0 ? 1 : -1',
        'binarize')

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class BinaryLinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = numpy.where(W>=0, 1, -1).astype(numpy.float32, copy=False)

        Xb = numpy.where(x>=0,1,-1).astype(x.dtype, copy=False)

        y = Xb.dot(Wb.T)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def forward_gpu(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = _kern()(W)

        Xb = _kern()(x)
        
        y = Xb.dot(Wb.T)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,


    def backward_cpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = numpy.where(W>=0, 1, -1).astype(numpy.float32, copy=False)
        gy = grad_outputs[0]

        gx = gy.dot(Wb).reshape(inputs[0].shape)
        gW = gy.T.dot(x)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW

    def backward_gpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = _kern()(W)
        gy = grad_outputs[0]

        gx = gy.dot(Wb).reshape(inputs[0].shape)
        gW = gy.T.dot(x)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def binary_linear(x, W, b=None):
    """Binary Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``..

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """
    if b is None:
        return BinaryLinearFunction()(x, W)
    else:
        return BinaryLinearFunction()(x, W, b)
