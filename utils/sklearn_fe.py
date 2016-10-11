from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn import metrics


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def custom_skfold(y, n_folds, random_state):
    try:
        return StratifiedKFold(y, n_folds=n_folds, random_state=random_state)
    except:
        try:
            return StratifiedKFold(y, n_folds=n_folds)
        except:
            return StratifiedKFold(y, k=n_folds)


def custom_kmeans(x, k, random_state, n_init=1, max_iter=50):
    try:
        return KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter).fit(x)
    except TypeError:
        return KMeans(k=k, random_state=random_state, n_init=n_init, max_iter=max_iter).fit(x)


def custom_gmm_score_samples(model, x):
    try:
        return model.score_samples(x)
    except:
        return model.eval(x)


def custom_accuracy_score(y_pred, y_true):
    try:
        return metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    except:
        return metrics.metrics.zero_one_score(y_pred=y_pred, y_true=y_true)