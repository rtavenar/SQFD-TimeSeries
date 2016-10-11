import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def kernel_normalize(k, kii=None, kjj=None):
    if kii is None:
        # Assuming k is self similarity matrix, then kii is the diagonal of k
        kii = numpy.diag(k)
        kjj = kii
    sqrt_kii = numpy.sqrt(kii).reshape((-1, 1))
    sqrt_kjj = numpy.sqrt(kjj).reshape((1, -1))
    k /= sqrt_kii
    k /= sqrt_kjj
    return k


def ideal_gram(labels):
    n = labels.shape[0]
    mat = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                mat[i, j] = 1
    return mat


def dist_ideal(gram, y):
    ideal_mat = ideal_gram(y)
    return numpy.linalg.norm(ideal_mat - gram)
