import tensorflow as tf

from modules.initializing import initializing
from modules.selecting import po
from modules.mechanism import check_learning_goal, restore_weights, save_weights, plot_residual
from modules.reorganizing import reorganizing
from modules.cramming import cramming
from modules.matching import matching


def tuning_weight(model, X_train, y_train, m, n, learning_goal, pdf):
    # save weights
    origin_weights = save_weights(model)

    # test
    # convert tf.tensor(n, 1) into array(n, )
    predict = model(X_train).numpy().reshape(-1, )
    residual = abs(predict - y_train)  # array(n, )

    # matching module
    result, matching_model = matching(model, X_train, y_train, epoch=150, optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01), learning_goal=learning_goal, show_result=True)

    if all(i <= 0.1 for i in residual[:5]) == False or all(i <= 10 for i in residual) == False:
        return result, matching_model
    else:
        # A
        if result:
            return result, matching_model  # acceptable SLFN
        # B
        else:
            # restore weights
            restore_model = restore_weights(
                matching_model, origin_weights, show_result=False)
            # cramming module
            cramming_model = cramming(
                restore_model, X_train, y_train, m, show_result=False)
            return result, cramming_model  # unacceptable SLFN


def learning_algorithm(trainnig, target, learning_goal=0.1):

    m = trainnig.shape[1]  # num of features
    big_N = trainnig.shape[0]  # num of rows

    n, initial_slfn = initializing(
        trainnig, target, m, big_N, show_result=True)  # initial n, initial model
    model = initial_slfn

    count = 0
    while n < big_N:  # stop criterion
        print("-----------------------Round {}------------------------------".format(n))
        # select proper trainning data
        inds = po(n)
        X_train, y_train = trainnig.iloc[inds].to_numpy(
        ), target.iloc[inds].to_numpy()

        # check if model reach the learning goal
        # if not, tune weight
        if check_learning_goal(model, X_train, y_train, learning_goal=learning_goal, show_result=True) == False:
            result, model = tuning_weight(
                model, X_train, y_train, m, n, learning_goal, pdf)

        # count matching acceptable
        if result:
            count = count+1

        # remove potentially irrelevant hidden nodes
        model = reorganizing(model, X_train, y_train,
                             learning_goal=learning_goal, show_result=False)

        # next run
        n = n+1

    print('acceptable times: {}'. format(count))
    return model
