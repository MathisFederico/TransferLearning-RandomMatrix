from keras.datasets import mnist
from sklearn.svm import LinearSVC
from evaluate import evaluate_model

# Choose a dataset (here the MNIST dataset)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Perform preproccesing here if needed
x_train = x_train.reshape((len(x_train), -1)) / 255.
x_test = x_test.reshape((len(x_test), -1)) / 255.

# Make a dataset object like so and give it a name
dataset = (x_train, y_train), (x_test, y_test)
dataset_name = 'MNIST'

# Set source and target labels for transfer learning (align, orthogonal will be computed automaticaly)
source_labels = [3, 4]
target_labels = [8, 9]

# Set your model and model name (needs a .predict method that provides classes labels prediction)
model = LinearSVC(loss='hinge', max_iter=2000)
model_name = 'SVM'

# Choose a number of experiment (the more the better for uncertainty !)
n_trials = 50
evaluate_model(model, n_trials, dataset, source_labels, target_labels,
                n_target=4, n_source_max=2000, points_per_trial=20,
                model_name=model_name, dataset_name=dataset_name,
                save_pickles=False)
