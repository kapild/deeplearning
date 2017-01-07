import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    b1 = np.zeros(hidden_dim)
    b2 = np.zeros(num_classes)
    W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
    W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b1"] = b1
    self.params["b2"] = b2
    self.params["W1"] = W1
    self.params["W2"] = W2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1 = self.params["W1"]
    B1 = self.params["b1"]
    W2 = self.params["W2"]
    B2 = self.params["b2"]
    reg = self.reg
    input_items = X.shape[0]
    out, cache = affine_relu_forward(X, W1, B1)
    scores, af_cache = affine_forward(out, W2, B2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    reg_loss = 0.5*reg * (np.sum(W1*W1) + np.sum(W2*W2))
    loss += reg_loss

    dout, dW2, dB2 = affine_backward(dx, af_cache)
    dX, dW1, dB1 = affine_relu_backward(dout, cache)

    # update the dw gradient with regularization loss
    dW2 += reg * W2    
    dW1 += reg * W1   

    grads["W2"] = dW2
    grads["b2"] = dB2
    grads["W1"] = dW1
    grads["b1"] = dB1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################


    for index in range(self.num_layers):
        left_dim = input_dim
        if index != 0:
            left_dim = hidden_dims[index - 1]
        if index == self.num_layers - 1:
            right_dim = num_classes
        else :
            right_dim = hidden_dims[index]
        self.params["W" + str(index + 1)] = weight_scale * np.random.randn(left_dim, right_dim)
        self.params["b" + str(index  + 1)] = np.zeros(right_dim)

        if use_batchnorm == True and index != self.num_layers -1 :
            self.params["gamma" + str(index + 1)] = np.ones(right_dim)
            self.params["beta" + str(index  + 1)] = np.zeros(right_dim)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    # this will store the cache of various layers of af_relu layer.
    cache_cache = {}

    reg = self.reg
    input_items = X.shape[0]
    out = X
    for index in range(self.num_layers - 1):
        w_b_layer = str(index + 1)
        w_affine_relu = self.params["W" + w_b_layer]
        b_affine_relu = self.params["b" + w_b_layer]
        if self.use_batchnorm:
            gamma = self.params["gamma" + w_b_layer]
            beta = self.params["beta" + w_b_layer]
            bn_param = self.bn_params[index]
            af_out, cache = affine_backprop_relu_forward(out, w_affine_relu, b_affine_relu, 
                                                         gamma, beta, bn_param)
        else:
            af_out, cache = affine_relu_forward(out, w_affine_relu, b_affine_relu)


        if self.use_dropout:
            af_out, dpout_cache = dropout_forward(af_out, self.dropout_param)
            cache = (cache, dpout_cache)

        cache_cache[w_b_layer] = cache
        out = af_out

    w_affine = self.params["W" + str(self.num_layers)]
    b_affine = self.params["b" + str(self.num_layers)]
    scores, affine_cache = affine_forward(out, w_affine, b_affine)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    reg_loss = 0.0
    # start from the last layer.
    for index in reversed(range(self.num_layers)):
        w = self.params["W" + str(index + 1)]
        reg_loss += (np.sum(w * w))

    loss += 0.5*reg * reg_loss

    # initially, update the weight for the last affine layer.
    dout, dW2, dB2 = affine_backward(dx, affine_cache)
    weight_bias_index = self.num_layers 
    weight_index_str = "W" + str(weight_bias_index)
    bias_index_str = "b" + str(weight_bias_index)
    dW2 += reg * self.params[weight_index_str]  
    grads[weight_index_str] = dW2
    grads[bias_index_str] = dB2

    affline_relu_dout = dout
    # run the backpropogation in reverse order for n - 1 layers.
    for index in reversed(range(self.num_layers - 1)):

        w_b_layer = str(index + 1)
        weight_index_str = "W" + str(w_b_layer)
        bias_index_str = "b" + str(w_b_layer)
        cache_cache_all = cache_cache[w_b_layer]
        # drop backward pass before relu and batch norm layer. 
        if self.use_dropout:
            dp_caches = cache_cache[w_b_layer]
            dropout_cache_index = len(dp_caches) - 1
            affline_relu_dout = dropout_backward(affline_relu_dout, dp_caches[dropout_cache_index])
            cache_cache_all = dp_caches[0:dropout_cache_index][0]

        if self.use_batchnorm:
            doutH, dWH, dBH, dgamma, dbeta = affine_back_prop_relu_backward(affline_relu_dout, cache_cache_all)
            grads["gamma" + str(w_b_layer)] = dgamma
            grads["beta" + str(w_b_layer)] = dbeta
        else:
            # import pdb
            # pdb.set_trace()

            doutH, dWH, dBH = affine_relu_backward(affline_relu_dout, cache_cache_all)
        affline_relu_dout = doutH
        dWH += reg * self.params[weight_index_str]  
        grads[weight_index_str] = dWH
        grads[bias_index_str] = dBH


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


def affine_backprop_relu_forward(x, W, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a backprop and ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, W, b)
  bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(bn_out)

  cache = (fc_cache, bn_cache, relu_cache)
  return relu_out, cache


def affine_back_prop_relu_backward(dout, cache):
  """
    Convenience layer that perorms an affine transform followed by a backprop and ReLU
  """
  (fc_cache, bn_cache, relu_cache) = cache
  dout_relu = relu_backward(dout, relu_cache)
  bn_out, dgamma, dbeta = batchnorm_backward(dout_relu, bn_cache)
  dx, dw, db =  affine_backward(bn_out, fc_cache)
  return dx, dw, db, dgamma, dbeta



def affine_backprop_forward(x, W, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform on batch norm

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, W, b)
  bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)

  cache = (fc_cache, bn_cache)
  return bn_out, cache



def affine_backprop_backward(dout, cache):
  (fc_cache, bn_cache) = cache
  bn_out, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
  dx, dw, db =  affine_backward(bn_out, fc_cache)
  return dx, dw, db, dgamma, dbeta


