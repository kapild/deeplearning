import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from fc_net import affine_backprop_relu_forward
from cnn import conv_batch_relu_forward, conv_batch_relu_backward
from fc_net import affine_back_prop_relu_backward, affine_backprop_forward, affine_backprop_backward


class DeepLayerConvNets(object):
  """
  A multi-layer convolutional network with the following architecture:
  
  [{conv - [batch norm] - relu - 2x2 max pool} * N - 1]  - [conv [batch norm] - relu] - [affine batch norm]xM - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  

  def __init__(self, input_dim=(3, 32, 32), 
               conv_layers_num_filters=[32, 32, 64, 64], 
               conv_layers_filter_size=[7, 7, 3, 3],
               affine_layers_hidden_dim=[100, 100, 100], num_classes=10, use_batchnorm=False, weight_scale=1e-3, reg=0.0,
               debugInit = False,
               dtype=np.float32):

    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - num_con_layers_n = number of deep layers for conv layer
    - num_affinee_layer_m = number of affine layer just before softmax.
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
    # TODO: Initialize weights and biases for the multi-layer convolutional    #
    # and multi layer affine network.                                          #
    # Weights should be initialized from a Gaussian with standard              #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim[0], input_dim[1], input_dim[2]
    # pass pool_param to the forward pass for the max-pooling layer
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # check if the filter size length equals number of filters per layer.
    assert(len(conv_layers_num_filters) == len(conv_layers_filter_size))

    n_cnn_layer, m_affine_layer = len(conv_layers_filter_size) - 1, len(affine_layers_hidden_dim)


    pool_size = self.pool_param['pool_height']
    pool_stride = self.pool_param['stride']
    
    ############################################################################
    # step 1 iterate and initialize N Conv layers.
    # All weights are of form W1-1, W1-2, ...  W1-n
    ############################################################################
    image_dim_depth = C
    image_dim_H = H
    image_dim_W = W
    if debugInit:
        print "Input Image layer 0:"
        print "\tInput Image H dim:" + str(image_dim_H)
        print "\tInput Image W dim:" + str(image_dim_W)
        print "\tInput Image depth dim:" + str(image_dim_depth)

    for index in range(n_cnn_layer):
        c1 = conv_layers_filter_size[index]
        f1 = conv_layers_num_filters[index]
        self.params["W1-" + str(index + 1)] = weight_scale * np.random.randn(f1, image_dim_depth, c1, c1)
        self.params["b1-" + str(index + 1)] = np.zeros(f1)
        # conv. batch norm
        if use_batchnorm == True:
            self.params["gamma1-" + str(index + 1)] = np.ones(f1)
            self.params["beta1-" + str(index + 1)] = np.zeros(f1)
        # no change in image size after conv layer as they are preserved.
        # only change happens after pooling using (W + 2 * P - pool_filter_size)/stride + 1
        # since padding is zero in pooling.
        image_dim_H = (image_dim_H - pool_size)/pool_stride + 1
        image_dim_W =  (image_dim_W - pool_size)/pool_stride + 1
        image_dim_depth = f1
        if debugInit:
            print "\n" + str(index + 1) + ":Conv. Relu Max Pool Layer" 
            print "\tInput Image H dim:" + str(image_dim_H)
            print "\tInput Image W dim:" + str(image_dim_W)
            print "\tInput Image depth dim:" + str(image_dim_depth)
            print "\tFilter Size:" + str(c1)
            print "\tNumber of filters:" + str(f1)

    ############################################################################
    # done init() N conv layers 
    ############################################################################

    
    ############################################################################
    # step 2 initialize single Conv. layers again
    ############################################################################
    c1 = conv_layers_filter_size[n_cnn_layer]
    f1 = conv_layers_num_filters[n_cnn_layer]
    self.params["W2"] = weight_scale * np.random.randn(f1, image_dim_depth, c1, c1)
    self.params["b2"] = np.zeros(f1)
    # conv. batch norm
    if use_batchnorm == True:
        self.params["gamma2"] = np.ones(f1)
        self.params["beta2"] = np.zeros(f1)
    if debugInit:
        print "\n" + str(n_cnn_layer + 1) + ":Conv Relu Layer" 
        print "\tInput Image H dim:" + str(image_dim_H)
        print "\tInput Image W dim:" + str(image_dim_W)
        print "\tInput Image depth dim:" + str(image_dim_depth)
        print "\tFilter Size:" + str(c1)
        print "\tNumber of filters:" + str(f1)

    ############################################################################
    # done wit second layer of conv . 
    ############################################################################
    
    ############################################################################
    # step 3 initialize M affline layers.
    ############################################################################
    left_dim = image_dim_H * image_dim_W * image_dim_depth
    for index in range(m_affine_layer):
        if index == m_affine_layer - 1:
            right_dim = num_classes
        else:
            right_dim = affine_layers_hidden_dim[index]

        self.params["W3-" + str(index + 1)] = weight_scale * np.random.randn(left_dim, right_dim)
        self.params["b3-" + str(index  + 1)] = np.zeros(right_dim)

        if use_batchnorm == True :
            self.params["gamma3-" + str(index + 1)] = np.ones(right_dim)
            self.params["beta3-" + str(index  + 1)] = np.zeros(right_dim)
        if debugInit:
            print "\n" + str(index + 1) + ":Affine Layer" 
            print "\tAffine left_dim dim:" + str(left_dim)
            print "\tAffine right_dim dim:" + str(right_dim)
        left_dim = right_dim

    ############################################################################
    # done init() M affine layers. 
    ############################################################################
    
    self.n_cnn_layer, self.m_affine_layer  = n_cnn_layer, m_affine_layer 
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    self.bn_params_affine = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(len(conv_layers_filter_size))]
      self.bn_params_affine = [{'mode': 'train'} for i in xrange(len(affine_layers_hidden_dim))]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as FullyConnectedNet in fc_net.py.
    """
    
    pool_param = self.pool_param
    mode = 'test' if y is None else 'train'

    import pdb
    # pdb.set_trace()
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
      for bn_param_affine in self.bn_params_affine:
        bn_param_affine[mode] = mode


    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the Deep convolutional net,         #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache_cache = {}
    cache_cache_affine = {}
    ############################################################################
    # 1. Forward pass for N -1  Conv. layers.
    ############################################################################
    out = X
    for index in range(self.n_cnn_layer):
        w_b_layer = str(index + 1)
        w_affine_relu = self.params["W1-" + w_b_layer]
        b_affine_relu = self.params["b1-" + w_b_layer]
        # pass conv_param to the forward pass for the convolutional layer
        # this change depening upon the filter size to keep image size preserved. (F-1)/2
        filter_size =  w_affine_relu.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        if self.use_batchnorm:
            gamma = self.params["gamma1-" + w_b_layer]
            beta = self.params["beta1-" + w_b_layer]
            bn_param = self.bn_params[index]
            out, layer1_cache = conv_backprop_relu_pool_forward(out, w_affine_relu, b_affine_relu, conv_param, 
                                                                pool_param, gamma, beta, bn_param)
        else:
            out, layer1_cache = conv_relu_pool_forward(out, w_affine_relu, b_affine_relu, conv_param, pool_param)
        cache_cache[w_b_layer] = layer1_cache

    ############################################################################
    # 2. Forward pass for single Conv. layer.
    ############################################################################
    index += 1
    w_affine_relu = self.params["W2"]
    b_affine_relu = self.params["b2"]
    filter_size =  w_affine_relu.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    if self.use_batchnorm:
        gamma = self.params["gamma2"]
        beta = self.params["beta2" ]
        bn_param = self.bn_params[index]
        out, layer1_cache = conv_batch_relu_forward(out, w_affine_relu, b_affine_relu, conv_param, 
                                                    gamma, beta, bn_param)
    else:
        out, layer1_cache = conv_relu_forward(out, w_affine_relu, b_affine_relu, conv_param)
    cache_conv_single_layer = layer1_cache

    ############################################################################
    # 3. Forward pass for M affline layers.
    ############################################################################
    for index in range(self.m_affine_layer):
        w_b_layer = str(index + 1)
        # print w_b_layer
        w_affine_relu = self.params["W3-" + w_b_layer]
        b_affine_relu = self.params["b3-" + w_b_layer]
        if self.use_batchnorm:
            gamma = self.params["gamma3-" + w_b_layer]
            beta = self.params["beta3-" + w_b_layer]
            bn_param_affine = self.bn_params_affine[index]
            af_out, cache = affine_backprop_forward(out, w_affine_relu, b_affine_relu, 
                                                         gamma, beta, bn_param_affine)
        else:
            af_out, cache = affine_forward(out, w_affine_relu, b_affine_relu)

        cache_cache_affine[w_b_layer] = cache
        out = af_out

    scores = out
    ############################################################################
    #                             END OF FORWARD PASS                          #
    ############################################################################


    # If test mode return early
    if mode == 'test':
      return scores


    ############################################################################
    #                             START OF BACKWARD PASS                       #
    ############################################################################
    
    loss, grads = 0, {}
    loss, dout = softmax_loss(scores, y)
    reg_loss = 0.0


    # start from the last layer.
    ############################################################################
    # 1. backward pass for M affline layers.
    ############################################################################
    reg = self.reg

    # # get regularized loss.
    # for index in reversed(range(self.m_affine_layer)):
    #     w_b_layer = str(index + 1)
    #     w = self.params["W3-" + w_b_layer]
    #     reg_loss += (np.sum(w * w))


    # run the backpropogation in reverse order for M layers.
    for index in reversed(range(self.m_affine_layer)):
        w_b_layer = str(index + 1)
        cache_cache_all = cache_cache_affine[w_b_layer]
        if self.use_batchnorm:
            dout, dWH, dBH, dgamma3, dbeta3 = affine_backprop_backward(dout, cache_cache_all)
            grads["gamma3-" + str(w_b_layer)] = dgamma3
            grads["beta3-" + str(w_b_layer)] = dbeta3
        else:
            dout, dWH, dBH = affine_backward(dout, cache_cache_all)
        # get regulaized loss, update gradients
        weight_index_str = "W3-" + str(w_b_layer)
        bias_index_str = "b3-" + str(w_b_layer)
        w = self.params[weight_index_str]
        dWH += reg * w  
        reg_loss += (np.sum(w * w))
        grads[weight_index_str] = dWH
        grads[bias_index_str] = dBH

    ############################################################################
    # 2. backward pass for Nth Conv batch norm Relu Layer.
    ############################################################################
    if self.use_batchnorm:
        dout, dW2, db2, dgamma2, dbeta2 = conv_batch_relu_backward(dout, cache_conv_single_layer)
        grads["gamma2"] = dgamma2
        grads["beta2"] = dbeta2
    else:
        dout, dW2, db2 = conv_relu_backward(dout, cache_conv_single_layer)

    w = self.params["W2"]  
    reg_loss += (np.sum(w * w))
    dW2 += reg * w
    grads["W2"] = dW2
    grads["b2"] = db2


    ############################################################################
    # 3. backward pass for (N-1) Conv batch norm relu max pool Layer.
    ############################################################################
    for index in reversed(range(self.n_cnn_layer)):
        w_b_layer = str(index + 1)
        layer1_cache = cache_cache[w_b_layer]     
        if self.use_batchnorm:
            dout, dW1, db1, dgamma1, dbeta1 = conv_backprop_relu_pool_backward(dout, layer1_cache)
            grads["gamma1-" + w_b_layer] = dgamma1
            grads["beta1-" + w_b_layer] = dbeta1
        else:
            dout, dW1, db1 = conv_relu_pool_backward(dout, layer1_cache)

        weight_index_str = "W1-" + str(w_b_layer)
        bias_index_str = "b1-" + str(w_b_layer)
        w = self.params[weight_index_str]
        reg_loss += (np.sum(w * w))
        dW1 += reg * w
        grads[weight_index_str] = dW1
        grads[bias_index_str] = db1
        


    loss += 0.5*reg * reg_loss

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


  



