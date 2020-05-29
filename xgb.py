from keras.datasets import mnist
from xgboost import XGBClassifier
from evaluate import evaluate_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((len(x_train), -1)) / 255.
x_test = x_test.reshape((len(x_test), -1)) / 255.

dataset = (x_train, y_train), (x_test, y_test)
source_labels = [3, 4]
target_labels = [8, 9]

model = XGBClassifier()
model_name = 'XGB'

n_trials = 50
evaluate_model(model, n_trials, dataset, source_labels, target_labels, n_target=4, n_source_max=2000, points_per_trial=20, model_name=model_name, save_pickles=False)
