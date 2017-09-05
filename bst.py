import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class BST(function.Function):

    """Binary with Straight Thourgh estimator Unit."""

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0]
        y = numpy.where(y>=0, 1, -1).astype(numpy.float32, copy=False)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x', 'T y',
            'y = x >= 0 ? 1 : -1', 'bst_fwd')(
                x[0])
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = abs(x) > 1 ? 0 : gy', 'bst_bwd')(
                x[0], gy[0])
        return gx,


def bst(x):
    """Binary with Straight Thourgh estimator Unit function.

    This function is expressed as

    .. math::
        f(x) = \\left \\{ \\begin{array}{ll}
        1 & {\\rm if}~ x \\ge 0 \\\\
        -1 & {\\rm if}~ x < 0,
        \\end{array} \\right.

    See: http://arxiv.org/abs/1511.07289

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return BST()(x)
