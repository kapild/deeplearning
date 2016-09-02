import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  num_feature = X.shape[1]

  sum_loss = 0.0
  for index in range(num_train):
    X_dot_W = X[index].dot(W)
    intermediate_scores = []
    for class_index in range(num_class):
      e_fyi = 0
      if class_index == y[index]:
        e_fyi = X_dot_W[class_index]   
      intermediate_scores.append(X_dot_W[class_index])
    intermediate_scores = np.array(intermediate_scores)      
    max_val = np.max(intermediate_scores)
    e_fyi -= max_val
    intermediate_scores -= max_val
    sum_loss += np.log(get_prob_at_index_k(y[index], intermediate_scores))
    # calculate gradient for dW  
    for class_index in range(num_class):
      # (yi - pk) * xi
      yi_minus_pk = - ((class_index == y[index]) - get_prob_at_index_k(class_index, intermediate_scores))
      factor = yi_minus_pk * X[index]
      dW[:, class_index] += factor

  loss = sum_loss/num_train + 0.5 * reg * np.sum(W * W) 

  loss = -1 * loss

  dW  = dW/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def get_prob_at_index_k(index_k, intermediate_scores):
  return np.exp(intermediate_scores[index_k])/np.sum(np.exp(intermediate_scores))

def softmax(z):
  sum_on_rows = np.sum(np.exp(z), axis = 1)
  return np.exp(z)/sum_on_rows[:, np.newaxis]

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  num_feature = X.shape[1]

  X_dot_W = X.dot(W)
  # get the max, this is done for numeric instability.
  X_dot_W_max = np.max(X_dot_W, axis = 1)
  # subtract the max from each row.
  X_dot_W_norm = X_dot_W - X_dot_W_max[:, np.newaxis]

  soft_vector = softmax(X_dot_W_norm)

  # get only the values of p(i) which has the correct class. 
  soft_vector_fired = soft_vector[(range(num_train), y)]
  sum_loss = np.sum(np.log(soft_vector_fired))
  loss = sum_loss/num_train + 0.5 * reg * np.sum(W * W) 
  loss = -1 * loss

  # start of gradient calculation, 
  # 1. we have to work in dimensions of N * C first.
  y_rows_matrix = np.zeros((num_train, num_class))
  y_rows_matrix[(range(num_train), y)] = 1

  pi_rows_matrix = soft_vector
  diff_yi_pi = pi_rows_matrix  - y_rows_matrix

  yi_pi_xi = X.transpose().dot(diff_yi_pi)
  dW  = yi_pi_xi/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW