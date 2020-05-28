
import numpy as np
np.random.seed(69)
import matplotlib.pyplot as plt

from keras.datasets import mnist
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from copy import copy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((len(x_train), -1)) / 255.
x_test = x_test.reshape((len(x_test), -1)) / 255.

source_labels = [3, 4]
orthogonal = False

if orthogonal:
    target_labels = [9, 8]
else:
    target_labels = [8, 9]

def split_classes(x, y, source, target):

    def in_source(x):
        return x in source_labels
    in_source = np.vectorize(in_source)

    def in_target(x):
        return x in target_labels
    in_target = np.vectorize(in_target)

    to_target = np.zeros(np.max(source + target))
    for i, src in enumerate(source):
        to_target[src] = target[i]

    x_source, y_source = x[in_source(y)], to_target[y[in_source(y)]]
    x_target, y_target = x[in_target(y)], y[in_target(y)]

    return (x_source, y_source), (x_target, y_target)


(x_source_train, y_source_train), (x_target_train, y_target_train) = split_classes(x_train, y_train, source_labels, target_labels)
(x_source_test, y_source_test), (x_target_test, y_target_test) = split_classes(x_test, y_test, source_labels, target_labels)

def select_n_per_class(n, x, y, classes, start_idx=0):
    x_new, y_new = [], []
    for cl in classes:
        x_new.append(copy(x[y==cl][start_idx:start_idx + n]))
        y_new.append(copy(y[y==cl][start_idx:start_idx + n]))
    return np.concatenate(x_new), np.concatenate(y_new)

n_source_max = int(2e3)
n_sources = np.unique(np.logspace(0, np.log10(n_source_max), num=100, dtype=int))
n_sources = np.concatenate(([0], n_sources))
n_target = 4
n_trials = 10

source_scores_global = []
target_scores_global = []
for trials in range(n_trials):

    print(f'  TRIAL {trials+1}')

    start_idx = np.random.randint(min(len(x_target_train), len(x_source_train)) // 3)

    source_scores = []
    target_scores = []
    for n_source in n_sources:
        svc = LinearSVC(loss='hinge', max_iter=2000)
        X_target_train, Y_target_train = select_n_per_class(n_target, x_target_train, y_target_train, target_labels, start_idx)
        if n_source > 0:
            X_source_train, Y_source_train = select_n_per_class(n_source, x_source_train, y_source_train, target_labels, start_idx)
            x_train = np.concatenate((X_source_train, X_target_train))
            y_train = np.concatenate((Y_source_train, Y_target_train))
            svc.fit(x_train, y_train)
        else:
            svc.fit(X_target_train, Y_target_train)
        source_score = svc.score(x_source_test, y_source_test)
        source_scores.append(source_score)
        target_score = svc.score(x_target_test, y_target_test)
        # print(f'{n_source} : {target_score}')
        target_scores.append(target_score)

    source_scores_global.append(np.array(source_scores))
    target_scores_global.append(np.array(target_scores))

source_scores_global = np.stack(source_scores_global, axis=1)
target_scores_global = np.stack(target_scores_global, axis=1)

source_scores_mean = np.mean(source_scores_global, axis=1)
target_scores_mean = np.mean(target_scores_global, axis=1)

source_scores_std = np.std(source_scores_global, axis=1)
target_scores_std = np.std(target_scores_global, axis=1)

fig, (ax1) = plt.subplots(1, 1)

ax1.semilogx(n_sources, source_scores_mean, color='b', label='SVM source_accuracy', linestyle=':')
ax1.semilogx(n_sources, target_scores_mean, color='b', label='SVM target_accuracy')

ax1.semilogx(n_sources, [target_scores_mean[0]]*len(n_sources), color='r', label='SVM without source', linestyle='--')

ax1.fill_between(n_sources, source_scores_mean - source_scores_std, source_scores_mean + source_scores_std, linewidth=0, color='b', alpha=0.1)
ax1.fill_between(n_sources, target_scores_mean - target_scores_std, target_scores_mean + target_scores_std, linewidth=0, color='b', alpha=0.1)

prefix = 'ortho' if orthogonal else 'align'
plt.title(f'Accuracy ({n_trials} trials) with {prefix} target')
plt.legend()
# ax1.set(xlim=(1, n_source_max), ylim=(0, 1))
plt.savefig(f'{prefix}.png')
plt.show()
