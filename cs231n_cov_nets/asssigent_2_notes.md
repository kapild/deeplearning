## Notes from assignemnt #2:


### Reshaping data

Reshape data to:`d_size` is a numebr here 

```
x_reshape = x.reshape(num_inputs, d_size)
```

Or you can reshape to an d1,d2,d3 stype array:

```
d_size = [1,2,3]
x_reshape = x.reshape(num_inputs, *d_size)
``` 

### Loss in backpropagation
loss with respect to any variable U is  nothing but 

```
dL/dU= 
where dL/dy * dy/da * da/du 
= dout * da/du
```

for U being W, the above equation becomes

```
dL/dw= 
where dL/dy * dy/da * da/dw 
= dout * da/dw

da/dw = np.dot(X,W)/dw 
which is X
```

### Loss function

Verify SoftMax and SVM loss equation. [here](https://github.com/kapild/deeplearning/blob/cs231n_cov_nets/cs231n_cov_nets/assignment2/cs231n/layers.py#L506)


## Add Parameters update

### Momemtum

### RMSPROP

### Adam



## Data Pre processing

### Zero Mean and center data: 
- Subtract mean image to each example
- 	Mean across 3 channels


### Weight Initilaization
- Never initalize weight to all ZERO.

- Init Use normal distribution:

	```
	0.01 np.random.randn(H1, H2)

	```
	- But, even with this stratget for 7-8 layers, the STD becomes zero. 
	- And, with vanishing gradient the dx in backpropagation is bad.

- Use `Xavier initilization`:
	
	```
	 np.random.randn(H1, H2)/ (np.sqrt(H1))

	```
- Xavier works with tanh(x) but doesn't work with Relu.

	```
	 np.random.randn(H1, H2)/ (np.sqrt(H1/2))

	```

### Hyperparameter ranges. 
- Search for hyperparameters on log scale. For example, a typical sampling of the learning rate would look as follows: 
```
	learning_rate = 10 ** uniform(-6, 1). 
```
- That is, we are generating a random number from a uniform distribution, but then raising it to the power of 10. 


### Batch normalization
- At training time, a batch normalization layer uses a minibatch of data to estimate the mean and standard deviation of each feature.
-  A running average of these means and standard deviations is kept during training, and at test time these running averages are used to center and normalize features.
-   the batch normalization layer includes learnable shift and scale parameters for each feature dimension: To un learn


-  Normalize via : zero mean and unit variance
	
	 ```
	 X = x - mean / np.sqrt(var (x) + eps)
	 ```
- Inserted after FullyConneted affine layer before Relu (non linear layer)
- But, you shift and scale and learn gamma and beta

	 ```
	 X = gamma * x_hat + beta
	 ```


- Check your training
	- 	Loss shoud be log 1/N with no regularization
	-  It should go up with regularization
	-  Overrfit data with small data set
	-  To find the `best learning rate`
		- If loss doesn't go down then learning rate.  1e-6
		- With e6 it goes to NaN so bad.: 1e6
		- Find a rough region in between using binary search

	- Coarse to fine strategy using small epoch
	
		``` java
			 10 ** uniform(-6, 6)
		```	 
	- Use random initilization

### Dropout. 
- Search for hyperparameters on log scale. For example, a typical sampling of the learning rate would look as follows: 
- higher = less dropout
- Train

	```
	  H1 = np.maximum(0, np.dot(W1, X) + b1)
	  U1 = np.random.rand(*H1.shape) < p 
	  H1 *= U1 # drop!
	```
- Test

	```
	  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activation
 	```
- Inverted droput
- Train

	```
	  U1 = (np.random.rand(*H1.shape) < p) / p
	```
