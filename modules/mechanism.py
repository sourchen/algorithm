import numpy as np
import matplotlib.pyplot as plt


def plot_residual(model, X_train, y_train, title=None):
    # convert tf.tensor(n, 1) into array(n, )
    predict = model(X_train).numpy().reshape(-1, )
    residual = abs(predict - y_train)  # array(n, )
    index = np.arange(len(y_train))

    fig = plt.figure()
    plt.plot(index, residual, 'ro')
    plt.title(title)
    plt.xlabel("Index of data")
    plt.ylabel("residual")
    plt.show()
    return fig


def save_weights(model):
    """
    save original weights

    :param model: current model
    :type model: 

    :rtype: list
    :return: original weights
    """
    return model.get_weights()


def restore_weights(model, weights, show_result=False):
    """
    restore original weights to model

    :param model: current model
    :type model: 
    :param weights: original weight
    :type weights: list

    :rtype: 
    :return: model with original weights
    """
    if show_result:
        print("-----------------------Restore weights------------------------------")
        print('restore original weights: {}'.format(weights))

    model.set_weights(weights)
    return model


def check_learning_goal(model, X_train, y_train, learning_goal=0.1, show_result=False):
    """
    do forward operation and
    check if trainnig data in model reach the learning goal

    :param model: current model
    :type model: 
    :param X_train: selected trainning data
    :type X_train: 2-d array
    :param y_train: selected target data
    :type y_train: 1-d array
    :param learning_goal: learning goal, default=0.5
    :type learning_goal: float
    :param show_result: default=True, print predicted and true target value
    :type learning_goal: Boolean

    :rtype: Boolean
    :return: if trainnig data in model reach the learning goal
    """
    predict = model(X_train).numpy().reshape(-1,
                                             )  # convert tf.tensor(n, 1) into array(n, )
    residual = abs(predict - y_train)  # array(n, )

    if show_result:
        print("predicted y: {}".format(predict))
        print("truth y: {}".format(y_train))
        print("learnig goal: {}".format(learning_goal))
        print("absolute residual: {}".format(np.round(residual, 3)))

    return all(i <= learning_goal for i in residual)
