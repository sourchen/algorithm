import numpy as np

def po(n):
    """
    Pick up the first n trainning data, and return the indices.

    :param n: the first n trainning data
    :type n: int

    :rtype: list
    :return: indices of selected data
    """

    indices = list(range(0, n))
    return indices


def LTS(current_model, trainnig, target, n):
    """ 
    Sort all N trainning data based upon the squared residual values, 
    then select the n trainning data that are the smallest ones among N squared residual value.

    :param current_model: model at n stage
    :type current_model: 
    :param trainning: all N trainning data
    :type trainning: DataFrame
    :param target: all N target data
    :type target: DataFrame
    :param n: at n stage
    :type n: int

    :rtype: list
    :return: indices of selected data
    """
    predict = current_model(trainnig)  # y_predict
    truth = target  # y_train

    # calculate squared residual values
    squared_residual = np.square(
        predict-truth).reshape(target.shape[0], )  # array(N, )

    # sort and select the n trainning data that are the smallest ones among N squared residual value
    sort_index = np.argsort(e)[:n]  # array(n,) #前n個data的index

    # return indices of selected data
    indices = sort_index.tolist()  # list
    return indices
