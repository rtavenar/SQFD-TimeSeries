import numpy
from scipy.spatial.distance import pdist, squareform
from sklearn import svm

from utils import sklearn_fe, kernel, dictionary, preprocess


def sqfd_sq(f, g, gamma_f, w_x=None, w_y=None):
    if w_x is None:
        w_x = 1. / f.shape[0] * numpy.ones((f.shape[0],))
    if w_y is None:
        w_y = 1. / g.shape[0] * numpy.ones((g.shape[0],))
    a = numpy.concatenate((f, g))
    m = numpy.exp(-gamma_f * squareform(pdist(a, "sqeuclidean")))
    w = numpy.concatenate((w_x, -w_y))
    return numpy.dot(numpy.dot(w, m), w)


def f2qf(features, d, k, random_state=None):
    km = sklearn_fe.custom_kmeans(features.reshape((-1, d)), k, random_state=random_state)
    centroids = km.cluster_centers_
    freq = numpy.bincount(km.predict(features.reshape((-1, d))))
    return centroids, freq.astype(numpy.float32) / numpy.sum(freq)


def f2qf_l(f_set, d, k, scale_t=None):
    if scale_t is None:
        scale_t = 1.
    n = f_set.shape[0]
    centroids = numpy.zeros((n, k, d))
    weights = numpy.zeros((n, k))
    for i in range(n):
        f_set_i = f_set[i].reshape((-1, d)).copy()
        f_set_i[:, 0] *= scale_t
        centroids[i], weights[i] = f2qf(f_set[i], d, k)
    return centroids, weights


def compute_features(l_features, k, scale_t=None, n_per_set=None, d=None):
    if n_per_set is None or d is None:
        assert not(n_per_set is None and d is None), "Either n_per_set or d should be given"
        if n_per_set is None:
            assert l_features.shape[1] % d == 0, "Dim. of stacked features should be divisible by d"
            n_per_set = l_features.shape[1] / d
        else:
            assert l_features.shape[1] % n_per_set == 0, "Dim. of stacked features should be divisible by n_per_set"
            d = l_features.shape[1] / n_per_set
    l_features = l_features.reshape((-1, n_per_set, d))
    if scale_t is not None:
        l_features = preprocess.rescale_dim0(l_features, scale_t)
    return f2qf_l(l_features, d, k, scale_t=scale_t)


def gram_matrix_noexp(centroids1, centroids2, w1, w2, gamma_f, self_similarity=False):
    kernel_matrix = numpy.zeros((centroids1.shape[0], centroids2.shape[0]))
    for i, set1 in enumerate(centroids1):
        for j, set2 in enumerate(centroids2):
            if j < i and self_similarity:
                kernel_matrix[i, j] = kernel_matrix[j, i]
            else:
                kernel_matrix[i, j] = sqfd_sq(set1, set2, gamma_f=gamma_f, w_x=w1[i], w_y=w2[j])
    return kernel_matrix


def gram_matrix(centroids1, centroids2, w1, w2, gamma_f, gamma_kernel, self_similarity=False):
    return numpy.exp(-gamma_kernel * gram_matrix_noexp(centroids1, centroids2, w1, w2, gamma_f,
                                                       self_similarity))


def gram_matrix_fun(gamma_f, gamma_kernel):
    return lambda centroids1, centroids2, w1, w2: gram_matrix(centroids1=centroids1, centroids2=centroids2, w1=w1,
                                                              w2=w2, gamma_f=gamma_f, gamma_kernel=gamma_kernel)


def gram_matrix_fun_noexp(gamma_f):
    return lambda centroids1, centroids2, w1, w2: gram_matrix_noexp(centroids1=centroids1, centroids2=centroids2, w1=w1,
                                                              w2=w2, gamma_f=gamma_f)


def gram_matrix_fun_sym(gamma_f, gamma_kernel):
    return lambda centroids, w: gram_matrix(centroids1=centroids, centroids2=centroids, w1=w, w2=w, gamma_f=gamma_f,
                                            gamma_kernel=gamma_kernel, self_similarity=True)


def gram_matrix_fun_sym_noexp(gamma_f):
    return lambda centroids, w: gram_matrix_noexp(centroids1=centroids, centroids2=centroids, w1=w, w2=w, gamma_f=gamma_f,
                                                  self_similarity=True)


def cv(x, y, d, gamma_features, gamma_kernel_values, k, C_values, gamma_times=None, n_folds=3, random_state=None):
    if gamma_times is None:
        gamma_times = [None]
    perfs = {}
    d2ideal = {}
    k_fold = sklearn_fe.custom_skfold(y=y, n_folds=n_folds, random_state=random_state)
    n_per_set = x.shape[1] / d
    assert k <= n_per_set
    for gamma_f in gamma_features:
        for gamma_t in gamma_times:
            scale_t = preprocess.gamma2scale(gamma_t, gamma_f)
            centroids, weights = compute_features(x, k, scale_t=scale_t, d=d)
            kernel_fun_noexp = gram_matrix_fun_sym_noexp(gamma_f)
            gram_noexp = kernel_fun_noexp(centroids, weights)
            for gamma_kernel in gamma_kernel_values:
                gram = numpy.exp(-gamma_kernel * gram_noexp)
                d2ideal[gamma_f, gamma_kernel, gamma_t] = kernel.dist_ideal(gram, y)
                for train_index, test_index in k_fold:
                    gram_train = gram[train_index, :][:, train_index]
                    gram_test = gram[test_index, :][:, train_index]
                    for C in C_values:
                        if (gamma_f, gamma_kernel, gamma_t, C) not in perfs.keys():
                            perfs[gamma_f, gamma_kernel, gamma_t, C] = []
                        clf = svm.SVC(C=C, kernel="precomputed")
                        clf.fit(gram_train, y[train_index])
                        y_pred = clf.predict(gram_test)
                        acc = sklearn_fe.custom_accuracy_score(y_pred=y_pred, y_true=y[test_index])
                        perfs[gamma_f, gamma_kernel, gamma_t, C].append(acc)
                print(gamma_f, gamma_t, gamma_kernel)
    for (gamma_f, gamma_kernel, gamma_t, C), values in perfs.items():
        perfs[gamma_f, gamma_kernel, gamma_t, C] = (numpy.mean(values), d2ideal[gamma_f, gamma_kernel, gamma_t])
    return dictionary.argmax_doubleval(perfs)
