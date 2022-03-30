import tensorflow as tf
from modules.mechanism import check_learning_goal

def matching(model, X_train, y_train, epoch=150, optimizer='adam', learning_goal=0.1, show_result=True):
    """
    adjust weight to obtain an SLFN

    :param model: current model
    :type model: 
    :param X_train: selected trainning data
    :type X_train: 2-d array
    :param y_train: selected target data
    :type y_train: 1-d array
    :param epoch: default = 100
    :type epoch: int
    :param optimizer: default = adam
    :type optimizer: 
    :param learning_goal: learning goal, default=0.05
    :type learning_goal: float

    :rtype: Boolean, 
    :return: if trainnig data in model reach the learning goal, 
    """
    if show_result:
        print("----------------------------Matching-------------------------")

    model.compile(optimizer=optimizer)
    loss_li = []
    epoch_li = []
    lr_li = []

    for i in range(epoch):
        # if show_result: print('/////////////////////////////////')
        # if show_result: print('Epoch: {}'.format(i+1))
        # forward operation
        if check_learning_goal(model, X_train, y_train, learning_goal=0.1):
            result = True
            acceptable_model = model  # acceptable SLFN

            if show_result:
                print('matching result: Acceptable SLFN')

            return result, acceptable_model
            break

        else:
            # backward operatin
            with tf.GradientTape() as tape:
                y_pred = model(X_train, training=True)
                y_pred = tf.reshape(y_pred, [y_pred.shape[0]])
                tape.watch(model.trainable_weights)
                tape.watch(y_pred)
                # Compute our own loss
                loss = tf.keras.losses.MeanSquaredError()(y_train, y_pred)
                loss_li.append(loss)
                epoch_li.append(i+1)
                lr_li.append(model.optimizer.learning_rate.numpy())
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_weights)
            # Update weights
            optimizer.apply_gradients(
                zip(gradients, model.trainable_weights))  # gradients

        if i == epoch-1:  # the last epoch
            result = False
            unacceptable_model = model  # acceptable SLFN

            if show_result:
                print('matching result: Unacceptable SLFN')

            return result, unacceptable_model
