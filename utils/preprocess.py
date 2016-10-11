import numpy
from scipy.spatial.distance import cdist
from scipy.stats import norm

from utils import numpy_fe

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def f_norm(l_features, n_per_set, d):
    lf_norm = l_features.reshape((-1, d))
    lf_norm /= numpy_fe.custom_norm(lf_norm, row_wise=True).reshape((-1, 1))
    return lf_norm.reshape((-1, n_per_set * d))


def rescale_dim0(data, scale):
    data_rs = data.copy()
    data_rs[:, :, 0] = scale * data_rs[:, :, 0]
    return data_rs


def gamma2scale(gamma_t, gamma_f):
    if gamma_t is None:
        return None
    else:
        return numpy.sqrt(gamma_t / gamma_f)


def gamma2norm(gamma_t, n_per_set):
    if gamma_t is None:
        return None
    else:
        m = cdist(numpy.arange(n_per_set).reshape((n_per_set, 1)) / n_per_set,
                  numpy.arange(n_per_set).reshape((n_per_set, 1)) / n_per_set, "sqeuclidean")
        # sqrt because we will do normalization for both phi vectors
        norm_factor = numpy.sqrt(numpy.sum(numpy.exp(-gamma_t * m)) / (n_per_set ** 2))
        return norm_factor


def gamma2approximate_norm(gamma_t):
    if gamma_t is None:
        return None
    else:
        g_t = gamma_t
        s_sq = numpy.sqrt(numpy.pi / g_t) * (2. * norm.cdf(numpy.sqrt(2. * g_t)) - 1.) - (1. - numpy.exp(-g_t)) / g_t
        return numpy.sqrt(s_sq)