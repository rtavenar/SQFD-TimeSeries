import sys
import numpy
from sklearn import svm, metrics

from datasets import UCRreader
from sets import sqfd_km
from utils import preprocess

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

Cs = numpy.logspace(0, 6, 5)
gamma_features = numpy.logspace(0, 6, 5)
gamma_kernel_values = numpy.logspace(-1, 5, 5)
gamma_times = [gamma_f for gamma_f in gamma_features]
normalize_features = False
use_time_info = True

target_dataset = sys.argv[1]
k = int(sys.argv[2])

if not use_time_info:
    gamma_times = [None]
path = "datasets/ucr_t/"
for ds_name in UCRreader.list_datasets(path=path):
    if target_dataset is None or target_dataset == ds_name:
        x_train, x_test, y_train, y_test, n_per_set, d = UCRreader.read_dataset_with_time(ds_name, path=path,
                                                                                normalize_features=normalize_features,
                                                                                          use_time_info=use_time_info)
        if k > n_per_set:
            k = n_per_set
        gamma_f, gamma_kernel, gamma_t, C = sqfd_km.cv(x_train, y_train, d, gamma_features, gamma_kernel_values, k, Cs,
                                                       gamma_times=gamma_times)
        kernel_fun = sqfd_km.gram_matrix_fun(gamma_f, gamma_kernel)
        kernel_fun_sym = sqfd_km.gram_matrix_fun_sym(gamma_f, gamma_kernel)
        scale_t = preprocess.gamma2scale(gamma_t, gamma_f)
        centroids_train, weights_train = sqfd_km.compute_features(x_train, k, scale_t=scale_t, n_per_set=n_per_set, d=d)
        centroids_test, weights_test = sqfd_km.compute_features(x_test, k, scale_t=scale_t, n_per_set=n_per_set, d=d)
        gram_train = kernel_fun_sym(centroids_train, weights_train)
        gram_test = kernel_fun(centroids_test, centroids_train, weights_test, weights_train)
        clf = svm.SVC(C=C, kernel="precomputed")
        clf.fit(gram_train, y_train)
        y_pred = clf.predict(gram_test)
        acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
        print("SQFD-KM_k%d;%s;%f;%f;%s;%f;%f" % (k, ds_name, gamma_f, gamma_kernel, str(gamma_t), C, 1 - acc))
