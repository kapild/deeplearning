# Tensor flow


## Variables

### Placeholder

```
tf.placeholder(tf.float32, [None, n])
tf.placeholder(tf.float32, [None, num_class])
tf.placeholder(tf.float32, [None])

```
### Variables

```
tf.Variable(tf.zeros([None, n]))

```

### Models

```
logits = tf.matmul(x, weights) + biases
tf.nn.softmax(logits)
```

--
### Cost function
```
cross_entropy  = tf.nn.softmax_cross_entropy(logits, labels)
cost = tf.reduce_mean(cross_entropy)
```

--
### Optimization function

```
tf.train.GradientDescentOptimizer(learning_rate).minimuze(cost)
```
--

## TensorFlow Run

```
session = tf.Session()
session.run(tf.initi_all_varisables())

data.train.next_batch(batch_size)

feed_dict_train = { x: , y_true: )

session.run(optimizer, feed_dict=feed_dict) 
```
--
#### one Hot coding 
```
one_hot=True
```

--


### Cov Nets.

- Normal weights

``` 
weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
```

- Conv. layers

```
new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling) 
shape = [filter_size, filter_size, channels, num_filters]
weights(shape)
biases(num_filters)

tf.nn.conv(input, filters, strides=[1,1,1,1], padding=Same)

```

- Pooling

```
tf.nn.max_pool(input, kszie, strides, paddig)
```
--
## Pretty Tensor.

```

with pt.defaults_scope(activation_fn = tt.nn.reul()):
	y_pred, loss = pt.wrap(image)
		.conv2d(kernel, depthm name)
		.max_pool(kernal=3, stride)
		.conv2d()
		.max_pool()

```

## Saver

```
saver = tf.train.Saver()
saver.save(sess=session, path)

```