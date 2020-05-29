# TransferLearning-RandomMatrix

This repository reproduce results found in :

R. Couillet, "A Random Matrix Analysis and Optimization Framework to Large Dimensional Transfer Learning"
https://romaincouillet.hebfree.org/docs/conf/transfer_learning.pdf

The framework is easy to use :

.. code-block::

    from keras.datasets import mnist
    from sklearn.svm import LinearSVC
    from evaluate import evaluate_model

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((len(x_train), -1)) / 255.
    x_test = x_test.reshape((len(x_test), -1)) / 255.
    
    dataset = (x_train, y_train), (x_test, y_test)
    dataset_name = 'MNIST'
    
    # Set source and target labels for transfer learning (align, orthogonal will be computed automaticaly)
    source_labels = [3, 4]
    target_labels = [8, 9]

    model = LinearSVC(loss='hinge', max_iter=2000)
    model_name = 'SVM'

    n_trials = 50
    evaluate_model(model, n_trials, dataset, source_labels, target_labels,
                    n_target=4, n_source_max=2000, points_per_trial=20,
                    model_name=model_name, dataset_name=dataset_name,
                    save_pickles=False)


This will save images like so :

.. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/MNIST/SVM/Accuracy%20(50%20trials)%20with%20align%20target.png
    :alt: Accuracy (50 trials) with align target
    :width: 700
    :align: center

.. image:: https://github.com/MathisFederico/TransferLearning-RandomMatrix/blob/master/images/MNIST/SVM/Accuracy%20(50%20trials)%20with%20ortho%20target.png
    :alt: Accuracy (50 trials) with orthogonal target
    :width: 700
    :align: center

