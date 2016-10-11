import sys
import numpy
from sklearn import svm, metrics

from datasets import UCRreader
from sets import bow

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

Cs = numpy.logspace(0, 6, 5)
ks = [2 ** i for i in range(4, 11)]
gamma_values = numpy.logspace(-4, 2, 5)
normalize_features = False
use_time_info = False

target_dataset = sys.argv[1]
n_coeff = int(sys.argv[2])


path = "datasets/ucr_t/"
for ds_name in UCRreader.list_datasets(path=path):
    if target_dataset is None or target_dataset == ds_name:
        x_train, x_test, y_train, y_test, n_per_set, d = UCRreader.read_dataset_with_time(ds_name, path=path,
                                                                                normalize_features=normalize_features,
                                                                                          use_time_info=use_time_info)
        k, gamma, C = bow.cv(x_train, y_train, d, ks, Cs, gamma_values=gamma_values, normalize=True)
        codebook = bow.learn_codebook(x_train, k, d)
        bows_train = bow.f_set2bow_set(x_train, codebook, normalize=True)
        bows_test = bow.f_set2bow_set(x_test, codebook, normalize=True)
        kernel_fun = bow.gram_matrix_fun(gamma=gamma)
        gram_train = kernel_fun(bows_train, bows_train)
        gram_test = kernel_fun(bows_test, bows_train)
        clf = svm.SVC(C=C, kernel="precomputed")
        clf.fit(gram_train, y_train)
        y_pred = clf.predict(gram_test)
        acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
        print("BoW;%s;%d;%s;%f;%f" % (ds_name, k, str(gamma), C, 1 - acc))
