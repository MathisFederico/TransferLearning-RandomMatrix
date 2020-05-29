from keras.datasets import mnist
from sklearn.svm import LinearSVC
from evaluate import evaluate_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((len(x_train), -1)) / 255.
x_test = x_test.reshape((len(x_test), -1)) / 255.

dataset = (x_train, y_train), (x_test, y_test)
source_labels = [3, 4]
target_labels = [8, 9]

model = LinearSVC(loss='hinge', max_iter=2000)
model_name = 'SVM'

n_trials = 5
evaluate_model(model, n_trials, dataset, source_labels, target_labels, n_target=4, n_source_max=2000, points_per_trial=20, model_name='SVM', save_pickles=False)
