import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import pickle
import time
import os

from sklearn.metrics import accuracy_score

def plot_evaluation(n_sources, source_scores_global, target_scores_global,
                    title=None, model_name='model', dataset_name=None,
                    save=True, show=False):
    
    def enlarge_std(x):
        """ Enlarge uncertainty at 95%"""
        n = len(x)
        unc = np.sqrt(np.std(x)/n)
        if n > 10:
            unc *= 2
        else:
            k = [0, 0, 12.7, 4.3, 3.18, 2.78, 2.57, 2.45, 2.37, 2.31, 2.26]
            unc *= k[n]
        return unc
    
    source_scores_mean = np.mean(source_scores_global, axis=1)
    target_scores_mean = np.mean(target_scores_global, axis=1)

    source_scores_unc = np.apply_along_axis(enlarge_std, 1, source_scores_global)
    target_scores_unc = np.apply_along_axis(enlarge_std, 1, target_scores_global)

    _, (ax1) = plt.subplots(1, 1)

    ax1.semilogx(n_sources, source_scores_mean, color='b', label=f'{model_name} source_accuracy', linestyle=':')
    ax1.semilogx(n_sources, target_scores_mean, color='b', label=f'{model_name} target_accuracy')
    ax1.semilogx(n_sources, [target_scores_mean[0]]*len(n_sources), color='r', label=f'{model_name} without source', linestyle='--')

    ax1.fill_between(n_sources, source_scores_mean - source_scores_unc, source_scores_mean + source_scores_unc, linewidth=0, color='b', alpha=0.1)
    ax1.fill_between(n_sources, target_scores_mean - target_scores_unc, target_scores_mean + target_scores_unc, linewidth=0, color='b', alpha=0.1)
    ax1.fill_between(n_sources, [target_scores_mean[0] - target_scores_unc[0]]*len(n_sources), [target_scores_mean[0] + target_scores_unc[0]]*len(n_sources), linewidth=0, color='r', alpha=0.1)
    
    ax1.set(xlim=(1, np.max(n_sources)), ylim=(0, 1))
    plt.legend()
    if title is not None:
        plt.title(title)

    if save:
        folder_path = os.path.join('images', model_name)
        if dataset_name is not None:
            folder_path = os.path.join(folder_path, dataset_name)
        os.makedirs(folder_path, exist_ok=True)

        figname = title if title else model_name + '_' + str(time.time())
        plt.savefig(os.path.join(folder_path, f'{figname}.png'))
    
    if show:
        plt.show()


def evaluate_model(model, n_trials, dataset, source_labels, target_labels,
                   n_target:int=4, n_source_max:int=2000, points_per_trial:int=100,
                   dataset_name=None, model_name='model', save_pickles=True):

    (x_train, y_train), (x_test, y_test) = dataset

    source_labels = np.array(source_labels)
    target_labels = np.array(target_labels)

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
    

    def select_n_per_class(n, x, y, classes, start_idx=0):
        x_new, y_new = [], []
        for cl in classes:
            x_new.append(copy(x[y==cl][start_idx:start_idx + n]))
            y_new.append(copy(y[y==cl][start_idx:start_idx + n]))
        return np.concatenate(x_new), np.concatenate(y_new)


    for orthogonal in (False, True):
        if orthogonal:
            target_labels = np.roll(target_labels, len(target_labels)//2, axis=0)

        (x_source_train, y_source_train), (x_target_train, y_target_train) = split_classes(x_train, y_train, source_labels, target_labels)
        (x_source_test, y_source_test), (x_target_test, y_target_test) = split_classes(x_test, y_test, source_labels, target_labels)

        n_sources = np.unique(np.logspace(0, np.log10(n_source_max), num=points_per_trial, dtype=int))
        n_sources = np.concatenate(([0], n_sources))

        prefix = 'ortho' if orthogonal else 'align'

        source_scores_global = []
        target_scores_global = []
        for trial in range(n_trials):
            x_target_train = np.random.permutation(x_target_train)
            y_target_train = np.random.permutation(y_target_train)

            source_scores = []
            target_scores = []
            for n_source in n_sources:
                model_transfer = deepcopy(model)
                X_target_train, Y_target_train = select_n_per_class(n_target, x_target_train, y_target_train, target_labels)
                if n_source > 0:
                    X_source_train, Y_source_train = select_n_per_class(n_source, x_source_train, y_source_train, target_labels)
                    X_train = np.concatenate((X_source_train, X_target_train))
                    Y_train = np.concatenate((Y_source_train, Y_target_train))
                    model_transfer.fit(X_train, Y_train)
                else:
                    model_transfer.fit(X_target_train, Y_target_train)

                y_source_pred = model_transfer.predict(x_source_test)
                source_score = accuracy_score(y_source_test, y_source_pred)
                source_scores.append(source_score)

                y_target_pred = model_transfer.predict(x_target_test)
                target_score = accuracy_score(y_target_test, y_target_pred)
                target_scores.append(target_score)

                print(f'Trial {trial+1} - {n_source} {prefix} sources - target_acc:{target_score}')

            source_scores_global.append(np.array(source_scores))
            target_scores_global.append(np.array(target_scores))

        source_scores_global = np.stack(source_scores_global, axis=1)
        target_scores_global = np.stack(target_scores_global, axis=1)

        if save_pickles:
            
            pickle_path = 'pickles'
            if dataset_name is not None:
                pickle_path = os.path.join(pickle_path, dataset_name)
            os.makedirs(pickle_path, exist_ok=True)
            
            with open(os.path.join(pickle_path, f'{model_name}_{n_trials}_{prefix}_nsources_{time.time()}.pkl'), 'wb') as f:
                pickle.dump(n_sources, f)
            with open(os.path.join(pickle_path, f'{model_name}_{n_trials}_{prefix}_source_{time.time()}.pkl'), 'wb') as f:
                pickle.dump(source_scores_global, f)
            with open(os.path.join(pickle_path, f'{model_name}_{n_trials}_{prefix}_target_{time.time()}.pkl'), 'wb') as f:
                pickle.dump(target_scores_global, f)

        title = f'Accuracy ({n_trials} trials) with {prefix} target'
        plot_evaluation(n_sources, source_scores_global, target_scores_global,
                        model_name=model_name, dataset_name=dataset_name, title=title,
                        save=True, show=False)
 

def load_and_plot(n_sources_path, source_scores_path, target_scores_path, save=True, show=True):
        with open(n_sources_path, 'rb') as f:
            n_sources = pickle.load(f)
        with open(source_scores_path, 'rb') as f:
            source_scores_global = pickle.load(f)
        with open(target_scores_path, 'rb') as f:
            target_scores_global = pickle.load(f)

        params = target_scores_path.split('_')
        model_name = params[0]
        n_trials = params[1]
        prefix = params[2]

        title = f'Accuracy ({n_trials} trials) with {prefix} target'
        plot_evaluation(n_sources, source_scores_global, target_scores_global, model_name=model_name, title=title, save=save, show=show)
 