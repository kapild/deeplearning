import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from fc_net import affine_backprop_relu_forward
from fc_net import affine_back_prop_relu_backward

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv [conv_batch norm]  - relu - 2x2 max pool - affine [batch norm]  - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, use_batchnorm=False, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim[0], input_dim[1], input_dim[2]
    pool = 2

    self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params["b1"] = np.zeros(num_filters)
    # conv. batch norm
    if use_batchnorm == True:
        self.params["gamma1"] = np.ones(num_filters)
        self.params["beta1"] = np.zeros(num_filters)

    w2_dimension = ((H - 2)/2 + 1) * ((W  -2)/2 + 1) * num_filters
    self.params["W2"] = weight_scale * np.random.randn(w2_dimension, hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim)
    # vanilla batch norm
    if use_batchnorm == True:
        self.params["gamma2"] = np.ones(hidden_dim)
        self.params["beta2"] = np.zeros(hidden_dim)

    self.params["b3"] = np.zeros(num_classes)
    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)    


    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]
    

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'

    import pdb
    # pdb.set_trace()
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # add batch norm for layer #1
    if self.use_batchnorm:
        # pdb.set_trace()
        gamma1 = self.params["gamma1"]
        beta1 = self.params["beta1"]
        gamma2 = self.params["gamma2"]
        beta2 = self.params["beta2"]
        # pdb.set_trace()

        layer1_out, layer1_cache = conv_backprop_relu_pool_forward(X, W1, 
            b1, conv_param, pool_param, gamma1, beta1, self.bn_params[0])
        layer2_out, layer2_cache = affine_backprop_relu_forward(layer1_out, W2, 
            b2, gamma2, beta2, self.bn_params[1])
    else:
        layer1_out, layer1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        layer2_out, layer2_cache = affine_relu_forward(layer1_out, W2, b2)

    scores, layer3_cache = affine_forward(layer2_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    reg_loss = 0.0
    # start from the last layer.
    reg_loss += (np.sum(W1 * W1))
    reg_loss += (np.sum(W2 * W2))
    reg_loss += (np.sum(W3 * W3))

    loss += 0.5*self.reg * reg_loss

    dout3, dW3, db3 = affine_backward(dx, layer3_cache)

    if self.use_batchnorm:
        dout2, dW2, db2, dgamma2, dbeta2 = affine_back_prop_relu_backward(dout3, layer2_cache)
        dout1, dW1, db1, dgamma1, dbeta1 = conv_backprop_relu_pool_backward(dout2, layer1_cache)
        grads["gamma2"] = dgamma2
        grads["beta2"] = dbeta2
        grads["gamma1"] = dgamma1
        grads["beta1"] = dbeta1

    else:
        dout2, dW2, db2 = affine_relu_backward(dout3, layer2_cache)
        dout1, dW1, db1 = conv_relu_pool_backward(dout2, layer1_cache)


    # add regulization
    dW3 += self.reg * self.params["W3"]
    dW2 += self.reg * self.params["W2"]
    dW1 += self.reg * self.params["W1"]

    grads["W3"] = dW3
    grads["W2"] = dW2
    grads["W1"] = dW1
    grads["b3"] = db3
    grads["b2"] = db2
    grads["b1"] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

def conv_backprop_relu_pool_forward(x, W, b, conv_param, pool_param, gamma, beta, bn_param):
  conv_a, conv_cache = conv_forward_fast(x, W, b, conv_param)
  # pdb.set_trace()
  out, spatial_cache = spatial_batchnorm_forward(conv_a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(out)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, spatial_cache, relu_cache, pool_cache)
  return out, cache



def conv_backprop_relu_pool_backward(dout, cache):
  conv_cache, spatial_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dsb, dgamma, dbeta = spatial_batchnorm_backward(da, spatial_cache)
  dx, dw, db = conv_backward_fast(dsb, conv_cache)
  return dx, dw, db, dgamma, dbeta


def conv_batch_relu_forward(x, W, b, conv_param, gamma, beta, bn_param):
  conv_a, conv_cache = conv_forward_fast(x, W, b, conv_param)
  out, spatial_cache = spatial_batchnorm_forward(conv_a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)
  cache = (conv_cache, spatial_cache, relu_cache)
  return out, cache



def conv_batch_relu_backward(dout, cache):
  conv_cache, spatial_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dsb, dgamma, dbeta = spatial_batchnorm_backward(da, spatial_cache)
  dx, dw, db = conv_backward_fast(dsb, conv_cache)
  return dx, dw, db, dgamma, dbeta


  




  



