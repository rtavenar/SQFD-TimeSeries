import numpy
from sklearn import svm
from scipy.spatial.distance import cdist

from utils import dictionary, kernel, sklearn_fe, numpy_fe

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def learn_codebook(features, k, d, random_state=None, sample=True):
    n = features.reshape((-1, d)).shape[0]
    if sample:
        indices = numpy.random.choice(n, size=min(100 * 1000, n), replace=False)
        sample_data = features.reshape((-1, d))[indices]
    else:
        sample_data = features.reshape((-1, d))
    return sklearn_fe.custom_kmeans(sample_data, k, random_state=random_state)


def f2bow(features, codebook):
    k, d = codebook.cluster_centers_.shape
    assign = codebook.predict(features.reshape((-1, d)))
    return numpy.bincount(assign, minlength=k)


def f_set2bow_set(l_features, codebook, normalize=False):
    bows = numpy.array([f2bow(set1, codebook) for set1 in l_features])
    if normalize:
        # SSR normalization
        bows = numpy.sqrt(bows)
        # L2 normalization
        bows /= numpy_fe.custom_norm(bows, row_wise=True).reshape((-1, 1))
    return bows


def gram_matrix(bows1, bows2, gamma=None):
    if gamma is not None:
        return numpy.exp(- gamma * cdist(bows1, bows2, "sqeuclidean"))
    return numpy.dot(bows1, bows2.T)


def gram_matrix_fun(gamma=None):
    return lambda set1, set2: gram_matrix(set1, set2, gamma)


def cv(x, y, d, k_values, C_values, gamma_values=None, n_folds=3, random_state=None, normalize=False):
    if gamma_values is None:
        gamma_values = [None]
    perfs = {}
    d2ideal = {}
    k_fold = sklearn_fe.custom_skfold(y=y, n_folds=n_folds, random_state=random_state)
    for k in k_values:
        n_per_set = x.shape[1] / d
        if k >= n_per_set * x.shape[0]:
            continue
        codebook = learn_codebook(x, k, d)
        bows = f_set2bow_set(x, codebook, normalize=normalize)
        for gamma in gamma_values:
            kernel_fun = gram_matrix_fun(gamma=gamma)
            gram = kernel_fun(bows, bows)
            d2ideal[k, gamma] = kernel.dist_ideal(gram, y)
            for train_index, test_index in k_fold:
                gram_train = gram[train_index, :][:, train_index]
                gram_test = gram[test_index, :][:, train_index]
                for C in C_values:
                    if (k, gamma, C) not in perfs.keys():
                        perfs[k, gamma, C] = []
                    clf = svm.SVC(C=C, kernel="precomputed")
                    clf.fit(gram_train, y[train_index])
                    y_pred = clf.predict(gram_test)
                    acc = sklearn_fe.custom_accuracy_score(y_pred=y_pred, y_true=y[test_index])
                    perfs[k, gamma, C].append(acc)
    for (k, gamma, C), values in perfs.items():
        perfs[k, gamma, C] = (numpy.mean(values), d2ideal[k, gamma])
    return dictionary.argmax_doubleval(perfs)
