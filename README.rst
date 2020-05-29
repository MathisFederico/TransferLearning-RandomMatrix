# TransferLearning-RandomMatrix

This repository reproduce results found in :

R. Couillet, "A Random Matrix Analysis and Optimization Framework to Large Dimensional Transfer Learning"
https://romaincouillet.hebfree.org/docs/conf/transfer_learning.pdf

The framework is easy to use :

.. code-block::

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


This will save images like so :

.. list-table:: 

    * - .. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/SVM/MNIST/Accuracy%20(50%20trials)%20with%20align%20target.png

      - .. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/SVM/MNIST/Accuracy%20(50%20trials)%20with%20ortho%20target.png

Here we can see that orthogonal (when transfer classes are badly done) is critical for the learning scheme.
Although alined features are beneficial for transfer learning, we can still see that:
1. Performences on the target is way less that for the source set
2. We need an order of magnitude more of sources elements to have the same influence on the target set than of the source set

Uncertainty is Type A with student enlargment. You can see details in evaluate.plot_evaluation().

Using a multilayer perceptron we have similar results:

.. list-table:: 

    * - .. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/MLP/MNIST/Accuracy%20(50%20trials)%20with%20align%20target.png

      - .. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/MLP/MNIST/Accuracy%20(50%20trials)%20with%20ortho%20target.png


Using xgboost we have similar results:

.. list-table:: 

    * - .. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/XGB/MNIST/Accuracy%20(50%20trials)%20with%20align%20target.png

      - .. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/XGB/MNIST/Accuracy%20(50%20trials)%20with%20ortho%20target.png




