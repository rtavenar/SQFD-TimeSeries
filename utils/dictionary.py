import numpy


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

EPS = 1e-9


def argmax_doubleval(d):
    k_opt = None
    maxval1, minval2 = -numpy.inf, numpy.inf
    for k, (v1, v2) in d.items():
        if (abs(v1 - maxval1) < EPS and v2 < minval2) or v1 >= maxval1 + EPS:
            k_opt = k
            maxval1 = v1
            minval2 = v2
    return k_opt
