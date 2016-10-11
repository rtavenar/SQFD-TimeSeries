import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def custom_norm(data, row_wise=False):
    if not row_wise:
        return numpy.linalg.norm(data)
    else:
        try:
            return numpy.linalg.norm(data, axis=1)
        except:
            n = data.shape[0]
            v = numpy.empty((n, ))
            for i in range(n):
                v[i] = numpy.linalg.norm(data[i])
            return v
