import tensorflow as tf
import numpy as np
import math


def random_r(m):
    """
    randomly create a m-vector r

    :param m: number of features
    :type m: integar
    :rtype: 2-d array
    :return: r, a m-vector
    """

    temp_r = np.random.randn(m,)  # randomly create a m-vector
    return temp_r.reshape(m, 1)  # shape = (m, 1)


def check_r(temp_r_transpose, trainning_data, small_number, m):
    """
    check if r is eligible for two condition

    :param temp_r_transpose: 2-d array
    :type temp_r_transpose: r, a m-vector
    :param trainning_data: selected n trainning data
    :type trainning_data: 2-d array
    :param small_number:
    :type small_number: float
    :param m: number of features
    :type m: integar

    :rtype: Boolean
    :return: if r is eligible for two condition
    """
    k = trainning_data.shape[0]  # k is a index

    check_list = []
    check_list = []

    # check each row
    for i in range(k-1):
        # for calculate matrix, reshape(m, ) into (m,1)
        each_row = (trainning_data[i]-trainning_data[k-1]).reshape(m, 1)
        x = temp_r_transpose.dot(each_row)[0][0]  # original [[x]]
        check_1 = (x != 0)
        check_2 = ((small_number+x) * (small_number-x) < 0)
        check_list.append((check_1 and check_2))

    return all(check_list)


def length_one(temp_r_transpose):
    """
    make m-vector r to lenght one 
    by dividing each component by its magnitude.

    :param temp_r_transpose: m-vector
    :type temp_r_transpose: 2-d array

    :rtype: 2-d array
    :return: r, a m-vector with length one
    """
    curren_len = math.sqrt(
        sum(np.square(temp_r_transpose.reshape(-1))))  # magnitude
    # dividing each component by its magnitude
    length_one_r = temp_r_transpose / curren_len
    return length_one_r


def create_r(trainning_data, small_number, m, show_result=True):  # trainning_data = c-k, k=n
    """
    create a eligible m-vector

    :param trainning_data: selected n trainning data
    :type trainning_data: 2-d array
    :param small_number: 
    :type small_number: float
    :param m: number of features
    :type m: integar

    :rtype: 2-d array
    :return: r, a m-vector with length one
    """
    k = trainning_data.shape[0]  # k is a index

    while True:
        # create r
        temp_r_transpose = random_r(m).T  # shape = (m, 1)

        # if r eligible
        if check_r(temp_r_transpose, trainning_data, small_number, m):
            # normalize a vector
            temp_r_transpose = length_one(temp_r_transpose)  # shape = (m, 1)
            if show_result:
                print("r: {0}, length is {1}".format(temp_r_transpose.T, math.sqrt(
                    sum(np.square(temp_r_transpose.reshape(-1))))))
            return temp_r_transpose.T
            break


def input_hidden_weight(r):
    """
    return new weights between input and hidden layer

    :param r: a eligible m-vector with lenght one
    :type r: 2-d array

    :rtype: array
    :return: weights between input and hidden layer
    """
    new_weights = np.concatenate((r, r), axis=1)
    new_weights = np.concatenate((new_weights, r), axis=1)

    return new_weights  # (m,3)


def hidden_bias(small_number, r, xk):  # k is array, shape = (m, )
    """
    use small_number, r, xk to calculate new bias,
    return new bias in hidden layer

    :param small_number: 
    :type small_number: float
    :param r: a eligible m-vector with lenght one
    :type r: 2-d array
    :param xk: the k trainning data
    :type xk: 1-d array

    :rtype: array
    :return: bias in hidden layer
    """
    rx = r.T.dot(xk)[0]  # constant
    new_bias = np.array([small_number-rx, -rx, (-small_number)-rx])
    return new_bias  # shape = (3, )


def hidden_output_weight(old_model, xk, yk, small_number):
    """
    use old_model, xk, yk, small_number to calculate new bias,
    return new weights between hidden and output layer

    :param old_model: current SLFN with p hidden nodes
    :type old_model:
    :param xk: the k trainning data
    :type xk: 1-d array
    :param yk: the k target data
    :type yk: float
    :param small_number: 
    :type small_number: float

    :rtype: 2-d array
    :return: weights between hidden and output layer
    """
    old_predict = old_model(xk.reshape(1, -1))
    residual = yk - old_predict  # TensorShape([1, 1])
    new_weights = np.array([(residual/small_number), -2 *
                           (residual/small_number), (residual/small_number)]).reshape(3, 1)
    return new_weights  # (3, 1)


def add_three_node(model, r, small_number, xk, yk, show_result=True):  # k is array, shape = (m, )
    """
    according r, small_number to obtain weights of three extra hidden nodes,
    add them to the existing SLFN

    :param model: selected n trainning data
    :type model: 2-d array
    :param r: m-vector 
    :type r: 2-d array
    :param small_number: 
    :type small_number: float
    :param xk: the k trainning data
    :type xk: 1-d array
    :param yk: the k target data
    :type yk: float

    :rtype: model
    :return: a new SLFN with p+3 hidden nodes
    """
    # Add 'units' in hidden layer config
    config = model.get_config()
    # edit three hidden node units
    config['layers'][1]['config']['units'] = config['layers'][1]['config']['units']+3
    # original weights
    weights = model.get_weights()

    if show_result:
        print("original weights")
        print(
            "- weights between input and hidden layers: {}".format(model.get_weights()[0]))
        print("- bias in hidden nodes: {}".format(model.get_weights()[1]))
        print(
            "- weights between hidden and output layers: {}".format(model.get_weights()[2]))
        print("- bias in output layer: {}".format(model.get_weights()[3]))

    # Add three hidden node weights
    # weight between input and hidden layers # (m, p) to (m, p+3)
    weights[0] = np.concatenate((weights[0], input_hidden_weight(r)), axis=1)
    weights[1] = np.append(weights[1], hidden_bias(
        small_number, r, xk))  # bias in hidden node
    weights[2] = np.concatenate((weights[2], hidden_output_weight(
        model, xk, yk, small_number)))  # weight between hidden and output layers

    # Create new model from the modified configuration, weights and return it
    new_model = tf.keras.Sequential.from_config(config)
    new_model.set_weights(weights)

    if show_result:
        print("new weights")
        print(
            "- weights between input and hidden layers: {}".format(new_model.get_weights()[0]))
        print("- bias in hidden nodes: {}".format(new_model.get_weights()[1]))
        print(
            "- weights between hidden and output layers: {}".format(new_model.get_weights()[2]))
        print("- bias in output layer: {}".format(new_model.get_weights()[3]))

    return new_model


# selected_trainning_data: array, shape(n, m), the last one is k(a array), selected_target: (n, 1)
def cramming(model, selected_trainning_data, selected_target, m, small_number=1e-05, show_result=True):
    """
    add three extra hidden nodes to the existing SLFN to obtain a new SLFN

    :param model: current SLFN with p hidden nodes
    :type model: 
    :param selected_trainning_data: selected n trainning data
    :type selected_trainning_data: 2-d array
    :param selected_target: selected n target data
    :type selected_target: 1-d array
    :param m: number of features
    :type m: integar
    :param small_number: default=1e-05
    :type small_number: float

    :rtype: 
    :return: a new SLFN with p+3 hidden nodes
    """
    if show_result:
        print("-----------------------Cramming------------------------------")

    # pick the k data in dataset
    xk = selected_trainning_data[-1]  # array, shape(m, )
    yk = selected_target[-1]  # constant
    if show_result:
        print("xk: {}".format(xk))
    if show_result:
        print("yk: {}".format(yk))

    # create an r vector
    r = create_r(selected_trainning_data, small_number, m,
                 show_result=show_result)  # shape = (m, 1)
    new_model = add_three_node(
        model, r, small_number, xk, yk, show_result=show_result)

    if show_result:
        print("--------------------------------------------------------------")

    return new_model
