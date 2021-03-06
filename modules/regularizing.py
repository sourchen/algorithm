# -*- coding: utf-8 -*-
"""regularizing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-aw9DNvvj5Gl5zjDwIGsbGafR1yj7Onr
"""

from algorithm.modules.mechanism import check_learning_goal, restore_weights, save_weights
import tensorflow as tf
from keras import backend as K
import numpy as np

def gradient(model, X_train, y_train, n, m, learning_rate_threshold):
  grad = True
  while grad:
    # backward operatin
    with tf.GradientTape() as tape: 
      y_pred = model(X_train, training=True) # Forward pass
      y_pred = tf.reshape(y_pred, [y_pred.shape[0]])
      tape.watch(model.trainable_weights)
      tape.watch(y_pred)
      # Compute original loss
      loss = tf.keras.losses.MeanSquaredError()(y_train, y_pred)# E(w)
      l2_loss = regularizing_loss(model, loss, n, m, decay_term=0.001)
          
    # Compute gradients
    gradients = tape.gradient(l2_loss, model.trainable_weights)
    # Update weights
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights)) # gradients

    # Compute loss after gradient
    y_pred_ = model(X_train, training=True)
    y_pred_ = tf.reshape(y_pred_, [y_pred_.shape[0]])
    loss_ = tf.keras.losses.MeanSquaredError()(y_train, y_pred_)# E(w')
    l2_loss_ = regularizing_loss(model, loss_, n, m, decay_term=0.001)

    if l2_loss_ <= l2_loss:
      return True, model
      break

    else:
      if (model.optimizer.lr > learning_rate_threshold)==False:
        return False, model
        break
      else:
        print('3')
        

def regularizing_loss(model, loss, n, m, decay_term=0.001):
  """
  calculate loss with L2 regularizing term in SLFN model

  :param model:
  :type model: 
  :param loss: model loss with loss function
  :type loss: float
  :param n: num of rows in training data
  :type n: 2-d array
  :param m: num of features in training data
  :type m: 2-d array
  :param decay_term: L2 regularizing term, default = 0.001
  :type decay_term: float
    
  :rtype: float
  :return: loss with L2 regularizing term
  """
  weights = model.get_weights()
  p = weights[0].shape[1]
  squared_weights = sum(np.square(weights[0].reshape(-1))) + sum(np.square(weights[1])) + sum(np.square(weights[2].reshape(-1))) + sum(np.square(weights[3]))
  total_loss = (loss/n) + (decay_term/(p+1+p*(m+1))) * squared_weights

  return total_loss

def slower_lr(model, show_result=False):
  """
  slow model's learning rate to 0.7 times

  :param model: original model
  :type model: 
  """
  if show_result: print("Learning rate slower: {}".format(model.optimizer.learning_rate.numpy()))
  lr = model.optimizer.learning_rate.numpy() * 0.7
  K.set_value(model.optimizer.learning_rate, lr)

def faster_lr(model, show_result=False):
  """
  fast model learning rate to 1.2 times

  :param model: original model
  :type model: 
  """
  if show_result: print("Learning rate faster: {}".format(model.optimizer.learning_rate.numpy()))
  lr = model.optimizer.learning_rate.numpy() * 1.2
  K.set_value(model.optimizer.learning_rate, lr)

def regularizing(model, X_train, y_train, epoch=100, optimizer='adam', learning_goal=0.05, learning_rate_threshold=0.0001, show_result=False):
  """
  reduce model weight magnitude

  :param model:
  :type model: 
  :param X_train: n trainning data
  :type X_train: 2-d array
  :param y_train: n target data
  :type y_train: 2-d array
  :type epoch: integar
  :param epoch: 
  :type optimizer: string
  :param optimizer: default = adam
  :type learning_goal: float
  :param learning_goal: default = 0.001
  :type learning_rate_threshold: float
  :param learning_rate_threshold: default = 0.0001
    
  :rtype: model
  :return: a new model, reduce model weight magnitude
  """
  current_model = model
  current_model.compile(optimizer=optimizer)

  m = X_train.shape[1]
  n = X_train.shape[0]

  for i in range(epoch):
    # save weights
    origin_weights = save_weights(current_model)

    grad_result, current_model = gradient(model, X_train, y_train, n, m, learning_rate_threshold=learning_rate_threshold)

    if grad_result:
      # if check_learning_goal(current_model, X_train[5:], y_train[5:], learning_goal=learning_goal, show_result=False):
      if check_learning_goal(current_model, X_train, y_train, learning_goal=learning_goal, show_result=False):
        faster_lr(current_model, show_result=show_result)
        print('1')
        # next epoch

      else:
        restore_weights(current_model, origin_weights, show_result=False)
        print('2')
        break
    
    else:
      restore_weights(current_model, origin_weights, show_result=False)
      print('4')
      break
    
  return current_model
