import sys
import numpy
from sklearn import svm, metrics
from sklearn.kernel_approximation import Nystroem

from datasets import UCRreader
from sets import sqfd_nystroem
from utils import preprocess

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

Cs = numpy.logspace(0, 6, 5)
gamma_features = numpy.logspace(0, 6, 5)
gamma_kernel_values = numpy.logspace(-1, 5, 5)
gamma_times = [gamma_f for gamma_f in gamma_features]
normalize_features = False
use_time_info = True

target_dataset = sys.argv[1]
n_coeff = int(sys.argv[2])

if not use_time_info:
    gamma_times = [None]
path = "datasets/ucr_t/"
for ds_name in UCRreader.list_datasets(path=path):
    if target_dataset is None or target_dataset == ds_name:
        x_train, x_test, y_train, y_test, n_per_set, d = UCRreader.read_dataset_with_time(ds_name, path=path,
                                                                                          normalize_features=normalize_features,
                                                                                          use_time_info=use_time_info)
        gamma_f, gamma_kernel, gamma_t, C = sqfd_nystroem.cv(x_train, y_train, d, gamma_features, gamma_kernel_values,
                                                             n_coeff, Cs, gamma_times=gamma_times, normalize=False)
        rbf_feature = Nystroem(gamma=gamma_f, n_components=n_coeff).fit(x_train.reshape((-1, d)))
        scale_t = preprocess.gamma2scale(gamma_t, gamma_f)
        phi_train = sqfd_nystroem.compute_features(x_train, rbf_feature, scale_t=scale_t, d=d, normalize=False)
        phi_test = sqfd_nystroem.compute_features(x_test, rbf_feature, scale_t=scale_t, d=d, normalize=False)
        clf = svm.SVC(C=C, kernel="rbf", gamma=gamma_kernel)
        clf.fit(phi_train, y_train)
        y_pred = clf.predict(phi_test)
        acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
        print("SQFD-Nystroem_d%d;%s;%f;%f;%s;%f;%f" % (n_coeff, ds_name, gamma_f, gamma_kernel, str(gamma_t), C, 1 - acc))
