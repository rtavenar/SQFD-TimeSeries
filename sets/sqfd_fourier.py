import numpy
from scipy.spatial.distance import cdist
from sklearn import svm, metrics
from sklearn.kernel_approximation import RBFSampler

from utils import kernel, dictionary, preprocess, numpy_fe, sklearn_fe


def phi_rbf_proj_set(set1, rbf_feature, norm_t=False):
    d = rbf_feature.n_components
    phi = numpy.mean(rbf_feature.transform(set1), axis=0)
    if norm_t is not None:
        phi /= norm_t
    return phi.reshape((d, ))



def gram_matrix(phi1, phi2, gamma_kernel):
    dist_matrix = cdist(phi1, phi2, "sqeuclidean")
    return numpy.exp(-gamma_kernel * dist_matrix)


def compute_features(l_features, rbf_feature, scale_t=None, n_per_set=None, d=None, normalize=False, norm_t=None):
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
    phi = numpy.array([phi_rbf_proj_set(set1, rbf_feature, norm_t=norm_t) for set1 in l_features])
    if normalize:
        # L2 normalization
        phi /= numpy_fe.custom_norm(phi, row_wise=True).reshape((-1, 1))
    return phi


def cv(x, y, d, gamma_features, gamma_kernel_values, n_coeff, C_values, gamma_times=None, normalize=False, n_folds=3,
       random_state=None):
    if gamma_times is None:
        gamma_times = [None]
    perfs = {}
    d2ideal = {}
    n_per_set = x.shape[1] / d
    k_fold = sklearn_fe.custom_skfold(y=y, n_folds=n_folds, random_state=random_state)
    for gamma_f in gamma_features:
        rbf_feature = RBFSampler(gamma=gamma_f, n_components=n_coeff).fit(x.reshape((-1, d)))
        for gamma_t in gamma_times:
            scale_t = preprocess.gamma2scale(gamma_t, gamma_f)
            norm_t = preprocess.gamma2norm(gamma_t, n_per_set=n_per_set)
            train_rff = compute_features(x, rbf_feature, scale_t=scale_t, d=d, normalize=normalize, norm_t=norm_t)
            for gamma_kernel in gamma_kernel_values:
                gram = gram_matrix(train_rff, train_rff, gamma_kernel)
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
                        acc = metrics.accuracy_score(y_pred=y_pred, y_true=y[test_index])
                        perfs[gamma_f, gamma_kernel, gamma_t, C].append(acc)
    for (gamma_f, gamma_kernel, gamma_t, C), values in perfs.items():
        perfs[gamma_f, gamma_kernel, gamma_t, C] = (numpy.mean(values), d2ideal[gamma_f, gamma_kernel, gamma_t])
    return dictionary.argmax_doubleval(perfs)
