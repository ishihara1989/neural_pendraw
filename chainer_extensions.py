from chainer import cuda
from chainer.functions.activation.softmax import softmax
import chainer.functions as F


def gumbel_softmax(log_pi, tau=0.1, axis=1):
    """Gumbel-Softmax sampling function.

    This function draws samples :math:`y_i` from Gumbel-Softmax distribution,
        :math:`y_i = {\\exp((g_i + \\log\\pi_i)/\\tau) \
                    \\over \\sum_{j}\\exp((g_j + \\log\\pi_j)/\\tau)}`,
        where :math:`\\tau` is a temperature parameter and \
            :math:`g_i` s are samples drawn from \
            Gumbel distribution :math:`Gumbel(0, 1)`

    See `Categorical Reparameterization with Gumbel-Softmax \
    <https://arxiv.org/abs/1611.01144>`_.

    Args:
        log_pi (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable representing pre-normalized
            log-probability :math:`\\log\\pi`.
        tau (:class:`~float`): Input variable representing \
            temperature :math:`\\tau`.

    Returns:
        ~chainer.Variable: Output variable.

    """
    xp = cuda.get_array_module(log_pi)
    dtype = log_pi.dtype
    g = xp.random.gumbel(size=log_pi.shape).astype(dtype)
    y = softmax((log_pi + g) / tau, axis=axis)

    return y


def leaky_clip(x, x_min, x_max, leak=0.2):
    return leak*x + (1-leak)*F.clip(x, x_min, x_max)
