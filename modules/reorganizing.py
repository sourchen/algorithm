import tensorflow as tf
import numpy as np

# import module
from modules.regularizing import regularizing
from modules.matching import matching


def delete_hidden_node(model, hidden_node_index):
    """
    delete hidden one node in model

    :param model: original model
    :type model: 
    :param hidden_node_index: index of the hidden node to be removed
    :type hidden_node_index: integar

    :rtype: model
    :return: a new model, remove one specified hidden node
    """
    # Reduce 'units' in hidden layer config
    config = model.get_config()
    # edit the hidden node units
    config['layers'][1]['config']['units'] = config['layers'][1]['config']['units']-1

    # Delete weights corresponding to deleted channels from config
    weights = model.get_weights()
    # delete the k column in weight between input and hidden layers
    weights[0] = np.delete(weights[0], hidden_node_index-1, axis=1)
    # delete the k bias in hidden node
    weights[1] = np.delete(weights[1], hidden_node_index-1)
    # delete the k column in weight between hidden and output layers
    weights[2] = np.delete(weights[2], hidden_node_index-1, axis=0)

    # Create new model from the modified configuration, weights and return it
    new_model = tf.keras.Sequential.from_config(config)
    new_model.set_weights(weights)
    return new_model


def reorganizing(model, X_train, y_train, learning_goal=0.05, show_result=True):
    """
    identify and remove the potentially irrelevant hidden node

    :param model: original model
    :type model: 
    :param X_train: n trainning data
    :type X_train: 2-d array
    :param y_train: n target data
    :type y_train: 2-d array

    :rtype: model
    :return: a new model, remove the irrelevant hidden node
    """
    if show_result:
        print("----------------------------Reorganizing-------------------------")

    k = 1
    p = model.get_weights()[0].shape[1]  # num of hidden nodes in current model
    if show_result:
        print("orginal num of hidden nodes: {}".format(p))
    current_model = model

    # examines all hidden nodes one by one
    while k <= p:
        if show_result:
            print('- Check the {0}th node in {1} hidden nodes'.format(k, p))

        # convert tf.tensor(n, 1) into array(n, )
        predict = current_model(X_train).numpy().reshape(-1, )
        residual = abs(predict - y_train)  # array(n, )

        # regularizing module
        current_model = regularizing(current_model, X_train, y_train, epoch=100, optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.05), learning_goal=learning_goal, learning_rate_threshold=0.0001)

        # convert tf.tensor(n, 1) into array(n, )
        predict = current_model(X_train).numpy().reshape(-1, )
        residual = abs(predict - y_train)  # array(n, )

        # temporarily ignore the k hidden node
        modified_model = delete_hidden_node(current_model, k)
        if show_result:
            print('- Temporarily ignore the {0}th node in {1} hidden nodes, current number of hidden nodes: {2}'.format(
                k, p, modified_model.get_weights()[0].shape[1]))

        # weight tuning module
        result, matching_model = matching(modified_model, X_train, y_train, epoch=150, optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.05), learning_goal=learning_goal, show_result=False)
        print('matching in reorganizing: {}'.format(result))
        # acceptable
        if result:
            p = p-1
            current_model = modified_model  # delete the hidden node
            if show_result:
                print('- Deleting the {0}th of {1} hidden nodes from hidden layer. Current number of hidden nodes: {2}'.format(
                    k, p, current_model.get_weights()[0].shape[1]))

        # unacceptable
        else:
            if show_result:
                print('- Kepp the {0}th node in {1} hidden nodes, current number of hidden nodes: {2}'.format(
                    k, p, current_model.get_weights()[0].shape[1]))
            # restore w
            k = k+1

        if show_result:
            print("----------------------------------------------------------------")
        if show_result:
            print('- Current num of hidden nodes: {}'.format(
                (np.array(current_model.get_weights()))[0].shape[1]))

    if show_result:
        print('- Final num of hidden nodes: {}'.format(
            (np.array(current_model.get_weights()))[0].shape[1]))
    return current_model
