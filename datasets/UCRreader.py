import numpy
import csv
import os

from utils import preprocess

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def read_file(fname, normalize=False):
    feature_sets = []
    cur_set = []
    for row in csv.reader(open(fname, "r"), delimiter=" "):
        if len(row) == 0:
            feature_sets.append(numpy.array(cur_set))
            cur_set = []
        else:
            cur_set.append(numpy.array([float(x) for x in row if x != ""]))
    if len(cur_set) > 0:
        feature_sets.append(numpy.array(cur_set))
    feature_sets = numpy.array(feature_sets)
    n_per_set, d = feature_sets.shape[1], feature_sets.shape[2]
    feature_sets = feature_sets.reshape((-1, n_per_set * d))
    if normalize:
        feature_sets = preprocess.f_norm(feature_sets, n_per_set, d)
    return feature_sets, n_per_set, d


def read_file_with_time(fname_mask, normalize=False, use_time_info=True):
    feature_sets = []
    i = 0
    while os.path.exists(fname_mask % i):
        data = numpy.loadtxt(fname_mask % i)
        if use_time_info:
            data[:, 0] /= numpy.max(data[:, 0])
        else:
            data = data[:, 1:]
        feature_sets.append(data)
        i += 1
    feature_sets = numpy.array(feature_sets)
    n_per_set, d = feature_sets.shape[1], feature_sets.shape[2]
    feature_sets = feature_sets.reshape((-1, n_per_set * d))
    if normalize:
        feature_sets = preprocess.f_norm(feature_sets, n_per_set, d)
    return feature_sets, n_per_set, d


def read_dataset(dataset_name, path="ucr/", normalize_features=False):
    data_train, n_per_set_train, d_train = read_file("%s%s/519/feature_vectors_0" % (path, dataset_name),
                                                     normalize=normalize_features)
    data_test, n_per_set_test, d_test = read_file("%s%s/519/feature_vectors_test_0" % (path, dataset_name),
                                                  normalize=normalize_features)
    assert n_per_set_test == n_per_set_train and d_train == d_test
    labels_train = numpy.loadtxt("%s/../ucr_classes/%s/train_classes" % (path, dataset_name)).astype(numpy.int32)
    labels_test = numpy.loadtxt("%s/../ucr_classes/%s/test_classes" % (path, dataset_name)).astype(numpy.int32)
    return data_train, data_test, labels_train, labels_test, n_per_set_train, d_train


def read_dataset_with_time(dataset_name, path="ucr_t/", normalize_features=False, use_time_info=True):
    data_train, n_per_set_train, d_train = read_file_with_time("%s%s/train_%%d" % (path, dataset_name),
                                                               normalize=normalize_features,
                                                               use_time_info=use_time_info)
    data_test, n_per_set_test, d_test = read_file_with_time("%s%s/test_%%d" % (path, dataset_name),
                                                            normalize=normalize_features,
                                                            use_time_info=use_time_info)
    assert n_per_set_test == n_per_set_train and d_train == d_test
    parent_path = parent_folder(path)
    labels_train = numpy.loadtxt("%s/ucr_classes/%s/train.txt" % (parent_path, dataset_name)).astype(numpy.int32)
    labels_test = numpy.loadtxt("%s/ucr_classes/%s/test.txt" % (parent_path, dataset_name)).astype(numpy.int32)
    return data_train, data_test, labels_train, labels_test, n_per_set_train, d_train


def parent_folder(path):
    if path[-1] == "/":
        path = path[:-1]
    if path.rfind("/") >= 0:
        return path[:path.rfind("/")]
    else:
        return path + "/.."


def list_datasets(path="ucr/"):
    l = []
    for p in os.listdir(path):
        if os.path.isdir(os.path.join(path, p)) and p != "ucr_classes":
            l.append(p)
    return l


if __name__ == "__main__":
    print(parent_folder("../datasets/ucr_t/"))
    # for dataset in list_datasets(path="ucr_t/"):
    #     data_train, data_test, labels_train, labels_test, n_per_set, d = read_dataset_with_time(dataset, path="ucr_t/")
    #     print(dataset, data_train.shape, data_test.shape, labels_train.shape, labels_test.shape, n_per_set, d,
    #           n_per_set * d)