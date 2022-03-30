import tensorflow as tf
import numpy as np


# trainning(n, m) # target(n, 1)

def calculated_data(trainning, target, indices, m, show_result=True):
    """ Calculate a set of m+1 weights by applying the linear regression on the dataset

    :param trainning: input data
    :type x: DataFrame
    :param target: target data
    :type y: DataFrame
    :param indices: indice of proper trainnig data
    :type indices: list

    :rtype: float, array (m+1, 1)
    :return: min value in indices of target, a set of m+1 weights
    """
    # slice data by indice, and turn to 2-array
    x = trainning.iloc[indices].to_numpy()  # array(m+1, m)
    y = target.iloc[indices].to_numpy(
    ) - target.iloc[indices].to_numpy().min()  # array(m+1, 1)

    # sovle m+1 linear equations in m+1 unknowns
    A = np.concatenate((np.ones((m+1, 1)), x), axis=1)  # matrix (m+1, m+1)
    B = y  # (m+1, 1)
    weigths = np.linalg.solve(A, B).reshape(m+1, 1)  # (m+1, 1)
    min_y = target.iloc[indices].to_numpy().min()

    if show_result:
        print("initial weights: {}".format(weigths))
        print("min y: {}".format(min_y))

    return min_y, weigths


def initial_weights(model, new_weights, min_y):  # (m+1, 1)
    """ assign initial weight to SLFN with one hidden node

    :param model: SLFN with one hidden node and m input nodes
    :type model:
    :param new_weights: initial weight
    :type new_weights: array (m+1m 1)
    :param min_y: min value of indice of y
    :type min_y: float
    """
    weights = model.get_weights()

    # w1~wm # weight between input and hidden layers
    weights[0] = new_weights[1:]
    weights[1] = new_weights[0]  # w0 # bias in hidden node
    weights[2] = np.array([[1]])  # weight between hidden and output layers
    weights[3] = np.array([min_y])

    # assign initial weights
    model.set_weights(weights)


def build_slfn(min_y, weights, m):
    """ Initiate a SLFN with one hidden node

    :param min_y: min value in indice of y
    :type min_y: float
    :param weights: calculated weight
    :type weights: array (m+1, 1)

    :rtype: float, array (m+1, 1)
    :return: min value in indices of target, a set of m+1 weights
    """
    # define model
    model = tf.keras.Sequential()
    # m input node, one hidden node
    model.add(tf.keras.layers.Dense(
        1, input_shape=(m,), activation=tf.nn.relu,))
    model.add(tf.keras.layers.Dense(1))  # one output node

    # assign initial weight
    initial_weights(model, weights, min_y)
    return model


def pick_data(m, big_N, show_result=True):
    """ pick up m+1 data (skip: that are linearly independent)

    :rtype: list 
    :return: a indice of trainning data  
    """
    indices = sorted(np.random.randint(0, big_N, size=(m+1)))

    print("pick up index of {} in trainnig data".format(indices))
    if show_result:
        print("pick up index of {} in trainnig data".format(indices))
    return indices


def initializing(x, y, m, big_N, show_result=True):
    """ Initializing module picks up proper trainig data to set up an acceptable SLFN with just one hidden node.

    :param x: input data
    :type x: DataFrame
    :param y: target data
    :type y: DataFrame

    :rtype: 
    :return: a SLFN with one hidden node     
    """
    print("----------------------------Initializing-------------------------" if show_result else '')

    # pick up m+1 data
    inds = pick_data(m, big_N, show_result=show_result)  # list

    if show_result:
        print("selected data in x: {}".format(x.iloc[inds].to_numpy()))
    if show_result:
        print("selected data in y: {}".format(y.iloc[inds].to_numpy()))

    # calculate a set of m+1 weights
    min_y, weights = calculated_data(x, y, inds, m, show_result=show_result)

    # set up the SLFN with one hidden node
    model = build_slfn(min_y, weights, m)

    if show_result:
        print("weights between input and hidden layers: {}".format(
            model.get_weights()[0]))
        print("bias in hidden nodes: {}".format(model.get_weights()[1]))
        print("weights between hidden and output layers: {}".format(
            model.get_weights()[2]))
        print("bias in output layer: {}".format(model.get_weights()[3]))

    n = m+2
    return n, model
